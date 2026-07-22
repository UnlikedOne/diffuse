use std::io::Write;
use std::time::{Duration, Instant};

use owo_colors::OwoColorize;

use diffuse_daemon::tui;

fn main() {
    println!();
    println!(
        "  {} {} {}",
        tui::human().truecolor(240, 200, 60),
        "›".truecolor(240, 200, 60).bold(),
        "who is michael jackson?".truecolor(230, 235, 245)
    );

    let seal = tui::phase_spinner("sealing prompt");
    std::thread::sleep(Duration::from_millis(400));
    seal.finish_and_clear();
    tui::phase_done("prompt sealed", "X25519 · ChaCha20");

    let route = tui::phase_spinner("routing through the network");
    std::thread::sleep(Duration::from_millis(400));
    route.finish_and_clear();
    tui::phase_done("routed", "1 encrypted hop");

    let think = tui::phase_spinner("thinking");
    std::thread::sleep(Duration::from_millis(900));
    think.finish_and_clear();

    tui::role(tui::robot(), "diffuse", tui::ACCENT, "");
    let rule = tui::sym("─", "-").repeat(46);
    println!("  {}", rule.truecolor(tui::FAINT.0, tui::FAINT.1, tui::FAINT.2));
    println!();
    print!("  ");
    let _ = std::io::stdout().flush();

    let mut live = tui::LiveMeter::new();
    let answer = "Michael Jackson was an American singer, songwriter, and dancer. He was born on August 29, 1958, and became one of the most influential figures in the history of popular music, earning the nickname King of Pop.";
    let tokens: Vec<String> = answer.split_inclusive(' ').map(|s| s.to_string()).collect();
    let start = Instant::now();
    let mut count = 0usize;

    for tok in tokens {
        live.push(&tok);
        count += 1;
        let rate = count as f64 / start.elapsed().as_secs_f64().max(0.001);
        live.tick(&tui::meter_text(rate, count));
        std::thread::sleep(Duration::from_millis(60));
    }

    live.clear();
    println!();
    let rate = count as f64 / start.elapsed().as_secs_f64().max(0.001);
    tui::hud(1, rate, 36);
    println!();
}
