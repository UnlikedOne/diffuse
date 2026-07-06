use owo_colors::OwoColorize;

use crate::capacity::ModelCapacity;

pub fn render_network_state(caps: &[ModelCapacity], node_count: usize, target_replication: usize) {
    println!();
    println!("{}", "  ◆ DIFFUSE NETWORK".bright_cyan().bold());
    println!(
        "  {} {} nodes online",
        "●".bright_green(),
        node_count.to_string().bright_white().bold()
    );
    println!();

    if caps.is_empty() {
        println!("  {}", "No models on the network yet.".dimmed());
        println!();
        return;
    }

    for cap in caps {
        let status = if cap.is_robust(target_replication) {
            "ROBUST".bright_green().bold().to_string()
        } else if cap.servable {
            "FRAGILE".yellow().bold().to_string()
        } else {
            "INCOMPLETE".red().bold().to_string()
        };

        println!(
            "  {} {}  [{}]",
            "▸".bright_blue(),
            cap.model_id.bright_white().bold(),
            status
        );

        let covered_layers: u32 = cap
            .slices
            .iter()
            .filter(|s| s.replicas > 0)
            .map(|s| s.end_layer.saturating_sub(s.start_layer))
            .sum();

        let bar = render_bar(covered_layers, cap.total_layers, 24);
        println!(
            "      {} {}/{} layers",
            bar,
            covered_layers.to_string().bright_white(),
            cap.total_layers.to_string().dimmed()
        );

        for s in &cap.slices {
            let rep_color = match s.replicas {
                0 => "○○○".red().to_string(),
                1 => "●○○".yellow().to_string(),
                _ => "●●●".bright_green().to_string(),
            };
            println!(
                "        {}:{}  {} {} replica(s)",
                s.start_layer.to_string().dimmed(),
                s.end_layer.to_string().dimmed(),
                rep_color,
                s.replicas
            );
        }
        println!();
    }
}

fn render_bar(current: u32, total: u32, width: usize) -> String {
    if total == 0 {
        return " ".repeat(width);
    }
    let filled = ((current as f64 / total as f64) * width as f64).round() as usize;
    let filled = filled.min(width);
    let bar: String = "█".repeat(filled) + &"░".repeat(width - filled);
    if current >= total {
        bar.bright_green().to_string()
    } else {
        bar.yellow().to_string()
    }
}