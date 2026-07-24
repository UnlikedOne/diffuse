use owo_colors::OwoColorize;

use crate::capacity::ModelCapacity;
use crate::tui;
use crate::worker::pb::ModelCard;
use crate::worker::WorkerHandle;

const SEARCH_SENTINEL: &str = "\u{0}search";
const PAGE_SIZE: usize = 12;

pub struct Listing {
    card: ModelCard,
    fits_whole: bool,
    peers_hosting: usize,
    servable: bool,
}

fn human_params(params: u64) -> String {
    if params == 0 {
        return "   ?  ".to_string();
    }
    let b = params as f64 / 1e9;
    if b >= 100.0 {
        format!("{:>5.0}B", b)
    } else if b >= 1.0 {
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

fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        format!("{:<width$}", text, width = width)
    } else {
        let kept: String = text.chars().take(width.saturating_sub(1)).collect();
        format!("{}…", kept)
    }
}

impl std::fmt::Display for Listing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.card.id == SEARCH_SENTINEL {
            return write!(
                f,
                "{}  {}",
                tui::sym("🔍", ">"),
                "search Hugging Face for another model".bold()
            );
        }

        let name = truncate(&self.card.id, 40);
        let size = human_params(self.card.params);

        let capacity = if self.card.params == 0 {
            "size unknown".truecolor(tui::MUTED.0, tui::MUTED.1, tui::MUTED.2).to_string()
        } else if self.fits_whole {
            format!("{} whole model", tui::sym("✓", "+"))
                .truecolor(tui::MINT.0, tui::MINT.1, tui::MINT.2)
                .to_string()
        } else {
            format!("{} as a slice", tui::sym("◐", "~"))
                .truecolor(tui::ACCENT.0, tui::ACCENT.1, tui::ACCENT.2)
                .to_string()
        };

        let network = if self.peers_hosting == 0 {
            "".to_string()
        } else if self.servable {
            format!("{} {} nodes, servable", tui::sym("●", "*"), self.peers_hosting)
                .truecolor(tui::MINT.0, tui::MINT.1, tui::MINT.2)
                .to_string()
        } else {
            format!("{} {} nodes, incomplete", tui::sym("◍", "o"), self.peers_hosting)
                .truecolor(tui::GOLD.0, tui::GOLD.1, tui::GOLD.2)
                .to_string()
        };

        let gated = if self.card.gated {
            format!(" {}", tui::sym("🔒", "[lock]"))
        } else {
            String::new()
        };

        let untested = if self.card.support == "likely" {
            format!(
                " {}",
                tui::sym("△", "!").truecolor(tui::YELLOW.0, tui::YELLOW.1, tui::YELLOW.2)
            )
        } else {
            String::new()
        };

        write!(
            f,
            "{} {} {:<24} {:>7} {}{}{}",
            name.bold(),
            size.truecolor(tui::TEXT.0, tui::TEXT.1, tui::TEXT.2),
            capacity,
            human_count(self.card.downloads)
                .truecolor(tui::FAINT.0, tui::FAINT.1, tui::FAINT.2),
            network,
            gated,
            untested,
        )
    }
}

fn build_listings(
    cards: Vec<ModelCard>,
    available_bytes: u64,
    caps: &[ModelCapacity],
) -> Vec<Listing> {
    let mut listings = vec![Listing {
        card: ModelCard {
            id: SEARCH_SENTINEL.to_string(),
            ..Default::default()
        },
        fits_whole: false,
        peers_hosting: 0,
        servable: false,
    }];

    for card in cards {
        let needed = card.params.saturating_mul(2);
        let fits_whole = card.params > 0 && (needed as f64) * 1.3 <= available_bytes as f64;
        let matching = caps.iter().find(|c| c.model_id == card.id);
        listings.push(Listing {
            fits_whole,
            peers_hosting: matching.map(|c| c.slices.len()).unwrap_or(0),
            servable: matching.map(|c| c.missing_slices().is_empty()).unwrap_or(false),
            card,
        });
    }
    listings
}

fn banner(device: &str, available_bytes: u64, count: usize) {
    tui::header(tui::sym("🛒", "#"), "marketplace");
    println!(
        "  {} {}   {} {}   {} hostable models",
        "machine:".truecolor(tui::MUTED.0, tui::MUTED.1, tui::MUTED.2),
        device.bright_white(),
        "free memory:".truecolor(tui::MUTED.0, tui::MUTED.1, tui::MUTED.2),
        human_bytes(available_bytes).bright_white(),
        count.to_string().bright_white(),
    );
    println!(
        "  {}",
        "type to filter, arrows to move, enter to host"
            .truecolor(tui::FAINT.0, tui::FAINT.1, tui::FAINT.2)
    );
    println!();
}

pub async fn browse(
    worker: &mut WorkerHandle,
    caps: &[ModelCapacity],
) -> anyhow::Result<Option<String>> {
    let mut query = String::new();

    loop {
        let message = if query.is_empty() {
            "loading the Hugging Face catalogue...".to_string()
        } else {
            format!("searching for {}...", query)
        };
        let spinner = tui::spinner(&message);
        let result = worker.search_models(&query, 40, true).await;
        spinner.finish_and_clear();

        let (cards, available_bytes, device) = result?;
        if cards.is_empty() {
            tui::warn(&format!("no hostable model matches {}", query));
            query.clear();
            continue;
        }

        banner(&device, available_bytes, cards.len());
        let listings = build_listings(cards, available_bytes, caps);

        let chosen = inquire::Select::new("model to host", listings)
            .with_page_size(PAGE_SIZE)
            .with_help_message("arrows to move, enter to host, type to filter, esc to quit")
            .prompt_skippable()?;

        let Some(chosen) = chosen else {
            return Ok(None);
        };

        if chosen.card.id == SEARCH_SENTINEL {
            let typed = inquire::Text::new("search:")
                .with_help_message("name, author, or model family")
                .prompt_skippable()?;
            match typed {
                Some(text) if !text.trim().is_empty() => query = text.trim().to_string(),
                _ => query.clear(),
            }
            continue;
        }

        return Ok(Some(chosen.card.id));
    }
}

pub async fn confirm_selection(
    worker: &mut WorkerHandle,
    model_id: &str,
    overhead: f64,
    caps: &[ModelCapacity],
) -> anyhow::Result<bool> {
    let spinner = tui::spinner(&format!("profiling {}...", model_id));
    let profile = worker.profile_model(model_id, overhead).await;
    spinner.finish_and_clear();

    let profile = match profile {
        Ok(p) => p,
        Err(e) => {
            tui::error(&format!("could not profile this model: {}", e));
            return Ok(false);
        }
    };

    tui::section(tui::sym("◆", ">"), model_id);
    println!(
        "    {} layers, ~{} per layer, {} available on {}",
        profile.total_layers.to_string().bright_white(),
        human_bytes(profile.avg_layer_bytes).bright_white(),
        human_bytes(profile.available_bytes).bright_white(),
        profile.device.bright_white(),
    );

    if profile.max_layers == 0 {
        tui::error("this machine cannot hold a single layer of this model");
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
                println!(
                    "    {} already servable on the network, you would add redundancy",
                    tui::sym("●", "*").truecolor(tui::MINT.0, tui::MINT.1, tui::MINT.2)
                );
            } else {
                let missing = gaps
                    .iter()
                    .map(|(s, e)| format!("{}:{}", s, e))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "    {} the network needs layers {}",
                    tui::sym("◍", "o").truecolor(tui::GOLD.0, tui::GOLD.1, tui::GOLD.2),
                    missing.bright_yellow().bold()
                );
            }
        }
        None => println!(
            "    {} nobody hosts this model yet, you would be the first",
            tui::sym("✦", "+").truecolor(tui::VIOLET.0, tui::VIOLET.1, tui::VIOLET.2)
        ),
    }
    println!();

    Ok(inquire::Confirm::new("host this model?")
        .with_default(true)
        .prompt_skippable()?
        .unwrap_or(false))
}
