use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;

pub const ACCENT: (u8, u8, u8) = (90, 210, 235);
pub const MINT: (u8, u8, u8) = (80, 220, 160);
pub const GOLD: (u8, u8, u8) = (245, 220, 130);
pub const VIOLET: (u8, u8, u8) = (176, 148, 250);
pub const TEXT: (u8, u8, u8) = (226, 232, 240);
pub const MUTED: (u8, u8, u8) = (128, 140, 158);
pub const FAINT: (u8, u8, u8) = (86, 98, 118);
pub const RED: (u8, u8, u8) = (236, 104, 116);
pub const YELLOW: (u8, u8, u8) = (240, 192, 96);

pub fn ascii_only() -> bool {
    if std::env::var_os("DIFFUSE_ASCII").is_some() {
        return true;
    }
    matches!(std::env::var("TERM").as_deref(), Ok("dumb"))
}

pub fn sym(fancy: &'static str, plain: &'static str) -> &'static str {
    if ascii_only() {
        plain
    } else {
        fancy
    }
}

pub fn human() -> &'static str {
    sym("●", "*")
}

pub fn robot() -> &'static str {
    sym("◆", ">")
}

fn term_width() -> usize {
    let w = console::Term::stdout().size().1 as usize;
    if w == 0 {
        80
    } else {
        w
    }
}

pub struct LiveMeter {
    active: bool,
    width: usize,
    rows: usize,
    col: usize,
}

impl Default for LiveMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveMeter {
    pub fn new() -> Self {
        use std::io::IsTerminal;
        LiveMeter {
            active: std::io::stdout().is_terminal(),
            width: term_width(),
            rows: 1,
            col: 2,
        }
    }

    pub fn push(&mut self, raw: &str) {
        use std::io::Write;
        let mut out = std::io::stdout();
        let _ = write!(out, "{}", raw.truecolor(GOLD.0, GOLD.1, GOLD.2));
        let _ = out.flush();
        for c in raw.chars() {
            if c == '\n' {
                self.rows += 1;
                self.col = 0;
            } else {
                self.col += 1;
                if self.col >= self.width {
                    self.rows += 1;
                    self.col = 0;
                }
            }
        }
    }

    pub fn tick(&self, status: &str) {
        use std::io::Write;
        if !self.active {
            return;
        }
        let mut out = std::io::stdout();
        let _ = write!(out, "\x1b7\x1b[{}A\r\x1b[2K  {}\x1b8", self.rows, status);
        let _ = out.flush();
    }

    pub fn clear(&self) {
        use std::io::Write;
        if !self.active {
            return;
        }
        let mut out = std::io::stdout();
        let _ = write!(out, "\x1b7\x1b[{}A\r\x1b[2K\x1b8", self.rows);
        let _ = out.flush();
    }
}

pub fn lock() -> &'static str {
    sym("🔒", "*")
}

fn spinner_ticks() -> &'static [&'static str] {
    if ascii_only() {
        &["|", "/", "-", "\\"]
    } else {
        &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    }
}

fn done_glyph() -> &'static str {
    sym("◆", "*")
}

fn dot() -> &'static str {
    sym("·", "-")
}

fn rule_glyph() -> &'static str {
    sym("─", "-")
}

pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(spinner_ticks()),
    );
    pb.set_message(msg.truecolor(150, 200, 225).to_string());
    pb.enable_steady_tick(Duration::from_millis(80));
    pb
}

pub fn phase_spinner(msg: &str) -> ProgressBar {
    spinner(msg)
}

pub fn phase_done(label: &str, detail: &str) {
    let padded = format!("{:<22}", label);
    println!(
        "  {} {}{}",
        done_glyph().truecolor(MINT.0, MINT.1, MINT.2),
        padded.truecolor(TEXT.0, TEXT.1, TEXT.2),
        detail.truecolor(MUTED.0, MUTED.1, MUTED.2)
    );
}

pub fn divider() {
    let line = rule_glyph().repeat(46);
    println!("  {}", line.truecolor(FAINT.0, FAINT.1, FAINT.2));
    println!();
}

pub fn role(icon: &str, name: &str, accent: (u8, u8, u8), meta: &str) {
    println!();
    let head = format!(
        "{} {}",
        icon.truecolor(accent.0, accent.1, accent.2),
        name.truecolor(accent.0, accent.1, accent.2).bold()
    );
    if meta.is_empty() {
        println!("  {}", head);
    } else {
        println!("  {}   {}", head, meta.truecolor(MUTED.0, MUTED.1, MUTED.2));
    }
}

pub fn meter_text(rate: f64, tokens: usize) -> String {
    let frames = spinner_ticks();
    let spin = frames[tokens % frames.len()];
    format!(
        "{} {}  {}  {}  {}",
        spin.truecolor(ACCENT.0, ACCENT.1, ACCENT.2),
        lock().truecolor(ACCENT.0, ACCENT.1, ACCENT.2),
        format!("{:.1} tok/s", rate).truecolor(MINT.0, MINT.1, MINT.2),
        dot().truecolor(FAINT.0, FAINT.1, FAINT.2),
        format!("{} tokens", tokens).truecolor(MUTED.0, MUTED.1, MUTED.2)
    )
}

pub fn hud(hops: usize, tok_per_s: f64, compute_pct: u32) {
    let net = 100u32.saturating_sub(compute_pct);
    let hop_word = if hops == 1 { "hop" } else { "hops" };
    let line = format!(
        "{} e2e {d} {} {} {d} {:.1} tok/s {d} compute {}% {d} net {}%",
        lock(),
        hops,
        hop_word,
        tok_per_s,
        compute_pct,
        net,
        d = dot(),
    );
    println!("  {}", line.truecolor(FAINT.0, FAINT.1, FAINT.2));
}

pub fn header(icon: &str, title: &str) {
    println!();
    println!(
        "  {}  {}  {}  {}",
        icon,
        "DIFFUSE".truecolor(ACCENT.0, ACCENT.1, ACCENT.2).bold(),
        dot().truecolor(FAINT.0, FAINT.1, FAINT.2),
        title.truecolor(TEXT.0, TEXT.1, TEXT.2)
    );
}

pub fn section(icon: &str, title: &str) {
    println!();
    println!(
        "  {} {}",
        icon,
        title.truecolor(ACCENT.0, ACCENT.1, ACCENT.2).bold()
    );
    println!();
}

pub fn step(icon: &str, label: &str, detail: &str) {
    if detail.is_empty() {
        println!(
            "  {} {}",
            icon,
            label.truecolor(TEXT.0, TEXT.1, TEXT.2)
        );
    } else {
        let padded = format!("{:<24}", label);
        println!(
            "  {} {}{}",
            icon,
            padded.truecolor(TEXT.0, TEXT.1, TEXT.2),
            detail.truecolor(MUTED.0, MUTED.1, MUTED.2)
        );
    }
}

pub fn ok(label: &str, detail: &str) {
    step(&done_glyph().truecolor(MINT.0, MINT.1, MINT.2).to_string(), label, detail);
}

pub fn note(text: &str) {
    println!("  {}", text.truecolor(MUTED.0, MUTED.1, MUTED.2));
}

pub fn warn(text: &str) {
    let icon = sym("⚠", "!");
    println!(
        "  {} {}",
        icon.truecolor(YELLOW.0, YELLOW.1, YELLOW.2),
        text.truecolor(TEXT.0, TEXT.1, TEXT.2)
    );
}

pub fn error(text: &str) {
    let icon = sym("✗", "x");
    println!(
        "  {} {}",
        icon.truecolor(RED.0, RED.1, RED.2),
        text.truecolor(TEXT.0, TEXT.1, TEXT.2)
    );
}

pub fn badge(text: &str, color: (u8, u8, u8)) -> String {
    format!("{}", format!(" {} ", text).truecolor(color.0, color.1, color.2))
}
