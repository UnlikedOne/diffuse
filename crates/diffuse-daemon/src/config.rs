use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
pub enum Mode {
    Private,
    Public,
}

#[derive(Debug, Parser)]
#[command(name = "diffuse", version, about = "Decentralized private LLM inference")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Analyze this machine and show which slice it would host
    Plan {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "http://127.0.0.1:50051")]
        worker: String,
        #[arg(long, default_value_t = 0.3)]
        overhead: f64,
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
    },
    /// Run a legacy two-worker demo generation (temporary)
    Demo {
        #[arg(long, value_delimiter = ',', default_value = "http://127.0.0.1:50051")]
        stage_a: Vec<String>,
        #[arg(long, value_delimiter = ',', default_value = "http://127.0.0.1:50052")]
        stage_b: Vec<String>,
        #[arg(long, default_value = "Qwen/Qwen2.5-0.5B-Instruct")]
        model: String,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, value_delimiter = ',')]
        spares: Vec<String>,
    },
    /// Join the network: profile, pick a slice, load it, announce, and serve
    Host {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "http://127.0.0.1:50051")]
        worker: String,
        #[arg(long, default_value = "127.0.0.1:9440")]
        listen: String,
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
        #[arg(long, default_value_t = 0.3)]
        overhead: f64,
        #[arg(long, default_value_t = false)]
        spawn_worker: bool,
    },
    /// Query the network: discover a servable model, route through it, generate
    Query {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
        #[arg(long, default_value_t = 80)]
        max_tokens: usize,
    },
    /// List models currently hosted on the network
    Models {
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
    },
    /// Interactive chat with a model on the network
    Chat {
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
        #[arg(long, default_value_t = false)]
        memory: bool,
    },
}