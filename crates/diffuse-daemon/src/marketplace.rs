use std::io;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, ExecutableCommand};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use crate::capacity::ModelCapacity;
use crate::worker::pb::ModelCard;
use crate::worker::WorkerHandle;

const ACCENT: Color = Color::Rgb(90, 210, 235);
const MINT: Color = Color::Rgb(80, 220, 160);
const GOLD: Color = Color::Rgb(245, 200, 110);
const VIOLET: Color = Color::Rgb(176, 148, 250);
const CORAL: Color = Color::Rgb(236, 118, 128);
const TEXT: Color = Color::Rgb(226, 232, 240);
const MUTED: Color = Color::Rgb(128, 140, 158);
const FAINT: Color = Color::Rgb(84, 96, 116);
const SURFACE: Color = Color::Rgb(24, 28, 38);
const SELECTED: Color = Color::Rgb(38, 46, 62);

pub struct Listing {
    pub card: ModelCard,
    fits_whole: bool,
    holdable_layers: u32,
    peers: usize,
    servable: bool,
}

impl Listing {
    fn runnable(&self) -> bool {
        self.card.support != "unsupported" && self.holdable_layers > 0
    }
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{:.1} {}", value, UNITS[unit])
}

fn human_params(params: u64) -> String {
    if params == 0 {
        return "   ?  ".into();
    }
    let b = params as f64 / 1e9;
    if b >= 1.0 {
        format!("{:>5.1}B", b)
    } else {
        format!("{:>5.0}M", params as f64 / 1e6)
    }
}

fn human_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.0}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        text.to_string()
    } else {
        let kept: String = text.chars().take(width.saturating_sub(1)).collect();
        format!("{}…", kept)
    }
}

struct App {
    query: String,
    listings: Vec<Listing>,
    state: ListState,
    available_bytes: u64,
    device: String,
    account: String,
    authenticated: bool,
    editing: bool,
    status: String,
}

impl App {
    fn selected(&self) -> Option<&Listing> {
        self.state.selected().and_then(|i| self.listings.get(i))
    }

    fn step(&mut self, delta: isize) {
        if self.listings.is_empty() {
            return;
        }
        let len = self.listings.len() as isize;
        let current = self.state.selected().unwrap_or(0) as isize;
        self.state.select(Some((current + delta).rem_euclid(len) as usize));
    }
}

fn banner(area: Rect, buf: &mut Buffer, app: &App) {
    Block::default().style(Style::default().bg(SURFACE)).render(area, buf);

    let title = Line::from(vec![
        Span::styled("  ▞▞  ", Style::default().fg(ACCENT)),
        Span::styled("DIFFUSE", Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
        Span::styled("  marketplace", Style::default().fg(MUTED)),
    ]);

    let account = if app.authenticated {
        Span::styled(
            format!(
                "signed in as {}",
                if app.account.is_empty() { "hugging face" } else { &app.account }
            ),
            Style::default().fg(MINT),
        )
    } else {
        Span::styled("anonymous, gated models stay hidden", Style::default().fg(GOLD))
    };

    let machine = Line::from(vec![
        Span::raw("  "),
        Span::styled(
            app.device.to_uppercase(),
            Style::default().fg(VIOLET).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ·  ", Style::default().fg(FAINT)),
        Span::styled(human_bytes(app.available_bytes), Style::default().fg(TEXT)),
        Span::styled(" free", Style::default().fg(MUTED)),
        Span::styled("  ·  ", Style::default().fg(FAINT)),
        account,
    ]);

    Paragraph::new(vec![Line::from(""), title, machine]).render(area, buf);
}

fn search_bar(area: Rect, buf: &mut Buffer, app: &App) {
    let border = if app.editing { ACCENT } else { FAINT };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(Span::styled(
            if app.editing { " searching hugging face " } else { " press / to search " },
            Style::default().fg(if app.editing { ACCENT } else { MUTED }),
        ));
    let inner = block.inner(area);
    block.render(area, buf);

    let shown = if app.query.is_empty() && !app.editing {
        Span::styled("most downloaded models", Style::default().fg(FAINT))
    } else {
        Span::styled(
            format!("{}{}", app.query, if app.editing { "▏" } else { "" }),
            Style::default().fg(TEXT),
        )
    };
    Paragraph::new(Line::from(vec![Span::raw(" "), shown])).render(inner, buf);
}

fn model_rows(app: &App) -> Vec<ListItem<'static>> {
    app.listings
        .iter()
        .map(|item| {
            let runnable = item.runnable();
            let name_style = if runnable {
                Style::default().fg(TEXT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(FAINT)
            };

            let (badge, badge_style) = if !runnable {
                ("cannot run", Style::default().fg(CORAL))
            } else if item.fits_whole {
                ("whole model", Style::default().fg(MINT))
            } else {
                ("as a slice", Style::default().fg(ACCENT))
            };

            let media: Vec<&str> = item
                .card
                .inputs
                .iter()
                .filter(|m| m.as_str() != "text")
                .map(|m| m.as_str())
                .collect();

            let mut spans = vec![
                Span::styled(format!(" {:<32}", truncate(&item.card.id, 32)), name_style),
                Span::styled(format!("{} ", human_params(item.card.params)), Style::default().fg(TEXT)),
                Span::styled(format!("{:<12}", badge), badge_style),
                Span::styled(format!("{:<12}", media.join("+")), Style::default().fg(VIOLET)),
                Span::styled(format!("{:>6}", human_count(item.card.downloads)), Style::default().fg(FAINT)),
            ];
            if item.card.gated {
                spans.push(Span::styled("  gated", Style::default().fg(GOLD)));
            }
            if item.peers > 0 {
                spans.push(Span::styled(
                    format!("  {} hosting", item.peers),
                    Style::default().fg(if item.servable { MINT } else { GOLD }),
                ));
            }
            ListItem::new(Line::from(spans))
        })
        .collect()
}

fn detail(area: Rect, buf: &mut Buffer, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(FAINT))
        .title(Span::styled(" model ", Style::default().fg(MUTED)));
    let inner = block.inner(area);
    block.render(area, buf);

    let Some(item) = app.selected() else {
        Paragraph::new("no model selected")
            .style(Style::default().fg(FAINT))
            .render(inner, buf);
        return;
    };

    let mut lines = vec![
        Line::from(Span::styled(
            truncate(&item.card.id, inner.width.saturating_sub(1) as usize),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
    ];

    let mut row = |label: &str, value: String, style: Style| {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<13}", label), Style::default().fg(MUTED)),
            Span::styled(value, style),
        ]));
    };
    row("architecture", item.card.architecture.clone(), Style::default().fg(TEXT));
    row("layers", item.card.layers.to_string(), Style::default().fg(TEXT));
    row("parameters", human_params(item.card.params).trim().into(), Style::default().fg(TEXT));
    row("takes", item.card.inputs.join(", "), Style::default().fg(VIOLET));
    row("returns", item.card.outputs.join(", "), Style::default().fg(VIOLET));
    row("likes", human_count(item.card.likes), Style::default().fg(FAINT));

    lines.push(Line::from(""));
    if !item.runnable() {
        lines.push(Line::from(Span::styled(
            "this machine cannot host it",
            Style::default().fg(CORAL).add_modifier(Modifier::BOLD),
        )));
    } else if item.fits_whole {
        lines.push(Line::from(vec![
            Span::styled("this machine holds ", Style::default().fg(MUTED)),
            Span::styled("the whole model", Style::default().fg(MINT).add_modifier(Modifier::BOLD)),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("this machine holds ", Style::default().fg(MUTED)),
            Span::styled(
                format!("{} of {} layers", item.holdable_layers, item.card.layers),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
        ]));
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(item.card.note.clone(), Style::default().fg(MUTED))));

    if item.peers > 0 {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            if item.servable {
                format!("{} nodes serve it, you would add redundancy", item.peers)
            } else {
                format!("{} nodes hold parts, it is incomplete", item.peers)
            },
            Style::default().fg(if item.servable { MINT } else { GOLD }),
        )));
    }

    Paragraph::new(lines).wrap(Wrap { trim: true }).render(inner, buf);
}

fn footer(area: Rect, buf: &mut Buffer, app: &App) {
    let keys = if app.editing {
        vec![
            Span::styled(" enter ", Style::default().fg(SURFACE).bg(ACCENT)),
            Span::styled(" search   ", Style::default().fg(MUTED)),
            Span::styled(" esc ", Style::default().fg(SURFACE).bg(FAINT)),
            Span::styled(" cancel", Style::default().fg(MUTED)),
        ]
    } else {
        vec![
            Span::styled(" up down ", Style::default().fg(SURFACE).bg(FAINT)),
            Span::styled(" move   ", Style::default().fg(MUTED)),
            Span::styled(" / ", Style::default().fg(SURFACE).bg(ACCENT)),
            Span::styled(" search   ", Style::default().fg(MUTED)),
            Span::styled(" enter ", Style::default().fg(SURFACE).bg(MINT)),
            Span::styled(" host it   ", Style::default().fg(MUTED)),
            Span::styled(" q ", Style::default().fg(SURFACE).bg(FAINT)),
            Span::styled(" quit", Style::default().fg(MUTED)),
        ]
    };
    let mut lines = vec![Line::from(keys)];
    if !app.status.is_empty() {
        lines.push(Line::from(Span::styled(app.status.clone(), Style::default().fg(GOLD))));
    }
    Paragraph::new(lines).render(area, buf);
}

fn draw(frame: &mut Frame, app: &mut App) {
    let chunks = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(3),
        Constraint::Min(6),
        Constraint::Length(2),
    ])
    .split(frame.area());

    banner(chunks[0], frame.buffer_mut(), app);
    search_bar(chunks[1], frame.buffer_mut(), app);

    let body =
        Layout::horizontal([Constraint::Percentage(63), Constraint::Percentage(37)]).split(chunks[2]);

    let list = List::new(model_rows(app))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(FAINT))
                .title(Span::styled(
                    format!(" {} models, live from hugging face ", app.listings.len()),
                    Style::default().fg(MUTED),
                )),
        )
        .highlight_style(Style::default().bg(SELECTED))
        .highlight_symbol("▍");
    frame.render_stateful_widget(list, body[0], &mut app.state);

    detail(body[1], frame.buffer_mut(), app);
    footer(chunks[3], frame.buffer_mut(), app);
}

fn build_listings(
    cards: Vec<ModelCard>,
    available_bytes: u64,
    caps: &[ModelCapacity],
) -> Vec<Listing> {
    cards
        .into_iter()
        .map(|card| {
            let needed = card.params.saturating_mul(2);
            let fits_whole = card.params > 0 && (needed as f64) * 1.3 <= available_bytes as f64;
            let per_layer = if card.layers > 0 && card.params > 0 {
                needed / card.layers as u64
            } else {
                0
            };
            let usable = (available_bytes as f64 * 0.7) as u64;
            let holdable = if per_layer == 0 {
                card.layers
            } else {
                ((usable / per_layer.max(1)) as u32).min(card.layers)
            };
            let matching = caps.iter().find(|c| c.model_id == card.id);
            Listing {
                fits_whole,
                holdable_layers: holdable,
                peers: matching.map(|c| c.slices.len()).unwrap_or(0),
                servable: matching.map(|c| c.missing_slices().is_empty()).unwrap_or(false),
                card,
            }
        })
        .collect()
}

pub async fn browse(
    worker: &mut WorkerHandle,
    caps: &[ModelCapacity],
) -> anyhow::Result<Option<String>> {
    let (cards, available_bytes, device, authenticated, account) =
        worker.search_models("", 60, true).await?;

    let mut app = App {
        query: String::new(),
        listings: build_listings(cards, available_bytes, caps),
        state: ListState::default(),
        available_bytes,
        device,
        account,
        authenticated,
        editing: false,
        status: String::new(),
    };
    if !app.listings.is_empty() {
        app.state.select(Some(0));
    }

    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let outcome = run(&mut terminal, &mut app, worker, caps).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    outcome
}

async fn run<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    worker: &mut WorkerHandle,
    caps: &[ModelCapacity],
) -> anyhow::Result<Option<String>> {
    loop {
        terminal.draw(|frame| draw(frame, app))?;

        if !event::poll(Duration::from_millis(120))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        if app.editing {
            match key.code {
                KeyCode::Esc => {
                    app.editing = false;
                    app.query.clear();
                }
                KeyCode::Enter => {
                    app.editing = false;
                    app.status = "searching hugging face...".into();
                    terminal.draw(|frame| draw(frame, app))?;
                    match worker.search_models(&app.query, 60, true).await {
                        Ok((cards, bytes, device, auth, account)) => {
                            app.available_bytes = bytes;
                            app.device = device;
                            app.authenticated = auth;
                            app.account = account;
                            app.listings = build_listings(cards, bytes, caps);
                            app.state
                                .select(if app.listings.is_empty() { None } else { Some(0) });
                            app.status = if app.listings.is_empty() {
                                format!("nothing matches {}", app.query)
                            } else {
                                String::new()
                            };
                        }
                        Err(e) => app.status = format!("search failed: {}", e),
                    }
                }
                KeyCode::Backspace => {
                    app.query.pop();
                }
                KeyCode::Char(c) => app.query.push(c),
                _ => {}
            }
            continue;
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return Ok(None),
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Ok(None),
            KeyCode::Char('/') => {
                app.editing = true;
                app.query.clear();
                app.status.clear();
            }
            KeyCode::Down | KeyCode::Char('j') => app.step(1),
            KeyCode::Up | KeyCode::Char('k') => app.step(-1),
            KeyCode::Enter => {
                let Some(item) = app.selected() else { continue };
                if !item.runnable() {
                    app.status = format!("{}: {}", item.card.id, item.card.note);
                    continue;
                }
                return Ok(Some(item.card.id.clone()));
            }
            _ => {}
        }
    }
}

pub async fn confirm_selection(
    worker: &mut WorkerHandle,
    model_id: &str,
    overhead: f64,
    caps: &[ModelCapacity],
) -> anyhow::Result<bool> {
    use owo_colors::OwoColorize;

    let spinner = crate::tui::spinner(&format!("profiling {}...", model_id));
    let profile = worker.profile_model(model_id, overhead).await;
    spinner.finish_and_clear();

    let profile = match profile {
        Ok(p) => p,
        Err(e) => {
            crate::tui::error(&format!("could not profile this model: {}", e));
            return Ok(false);
        }
    };

    crate::tui::section(crate::tui::sym("◆", ">"), model_id);
    println!(
        "    {} layers, ~{} per layer, {} available on {}",
        profile.total_layers.to_string().bright_white(),
        human_bytes(profile.avg_layer_bytes).bright_white(),
        human_bytes(profile.available_bytes).bright_white(),
        profile.device.bright_white(),
    );

    if profile.max_layers == 0 {
        crate::tui::error("this machine cannot hold a single layer of this model");
        return Ok(false);
    }

    let share = (profile.max_layers as f64 / profile.total_layers.max(1) as f64 * 100.0).min(100.0);
    println!(
        "    this machine can hold {} layers ({:.0}% of the model)",
        profile.max_layers.to_string().bright_green().bold(),
        share,
    );

    match caps.iter().find(|c| c.model_id == model_id) {
        Some(cap) => {
            let gaps = cap.coverage_gaps();
            if gaps.is_empty() {
                println!("    {} already servable, you would add redundancy", "●".bright_green());
            } else {
                let missing = gaps
                    .iter()
                    .map(|(s, e)| format!("{}:{}", s, e))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "    {} the network needs layers {}",
                    "◍".bright_yellow(),
                    missing.bright_yellow().bold()
                );
            }
        }
        None => println!("    {} nobody hosts this model yet", "✦".bright_magenta()),
    }
    println!();

    Ok(inquire::Confirm::new("host this model?")
        .with_default(true)
        .prompt_skippable()?
        .unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn card(id: &str, params: u64, layers: u32, inputs: &[&str], support: &str) -> ModelCard {
        ModelCard {
            id: id.into(),
            architecture: "Qwen2ForCausalLM".into(),
            model_type: "qwen2".into(),
            params,
            downloads: 2_100_000,
            likes: 1432,
            gated: id.contains("meta-llama"),
            support: support.into(),
            note: "a plain decoder stack, splits cleanly".into(),
            layers,
            hidden_size: 3584,
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: vec!["text".into()],
        }
    }

    fn sample_app() -> App {
        let cards = vec![
            card("Qwen/Qwen2.5-0.5B-Instruct", 494_000_000, 24, &["text"], "ready"),
            card("HuggingFaceTB/SmolVLM-256M-Instruct", 256_000_000, 30, &["text", "image"], "ready"),
            card("Qwen/Qwen2-VL-2B-Instruct", 2_200_000_000, 28, &["text", "image", "video"], "ready"),
            card("mistralai/Voxtral-Mini-3B-2507", 4_680_000_000, 30, &["text", "audio"], "ready"),
            card("meta-llama/Llama-3.1-70B", 70_000_000_000, 80, &["text"], "ready"),
            card("openai/whisper-small", 244_000_000, 12, &["text"], "unsupported"),
        ];
        let mut app = App {
            query: String::new(),
            listings: build_listings(cards, 12_000_000_000, &[]),
            state: ListState::default(),
            available_bytes: 12_000_000_000,
            device: "cpu".into(),
            account: "unlikedone".into(),
            authenticated: true,
            editing: false,
            status: String::new(),
        };
        app.state.select(Some(2));
        app
    }

    fn render(app: &mut App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| draw(frame, app)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn the_marketplace_shows_what_each_model_takes_and_whether_it_fits() {
        let mut app = sample_app();
        let screen = render(&mut app, 118, 26);
        println!("{}", screen);

        assert!(screen.contains("DIFFUSE"), "the banner must be there");
        assert!(screen.contains("unlikedone"), "a signed in account is named");
        assert!(screen.contains("live from hugging face"), "the list says where it came from");
        assert!(screen.contains("whole model"));
        assert!(screen.contains("as a slice"));
        assert!(screen.contains("cannot run"));
        assert!(screen.contains("image+video"), "video capability must show");
        assert!(screen.contains("audio"));
    }

    #[test]
    fn an_anonymous_session_says_gated_models_are_hidden() {
        let mut app = sample_app();
        app.authenticated = false;
        app.account.clear();
        let screen = render(&mut app, 118, 26);
        assert!(screen.contains("anonymous"), "{}", screen);
    }

    #[test]
    fn searching_swaps_the_hint_for_what_is_typed() {
        let mut app = sample_app();
        app.editing = true;
        app.query = "qwen".into();
        let screen = render(&mut app, 118, 26);
        assert!(screen.contains("searching hugging face"));
        assert!(screen.contains("qwen"));
    }

    #[test]
    fn a_narrow_terminal_still_renders() {
        let mut app = sample_app();
        let screen = render(&mut app, 80, 20);
        assert!(screen.contains("DIFFUSE"));
    }
}
