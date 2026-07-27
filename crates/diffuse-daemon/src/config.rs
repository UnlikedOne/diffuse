use clap::{Parser, Subcommand, ValueEnum};

pub const DEFAULT_SENTINELS: &[&str] = &["http://204.168.151.107:9440"];

pub fn resolve_sentinels(provided: &[String]) -> Vec<String> {
    if provided.is_empty() {
        DEFAULT_SENTINELS.iter().map(|s| s.to_string()).collect()
    } else {
        provided.to_vec()
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum Mode {
    Private,
    Public,
}

#[derive(Debug, Parser)]
#[command(name = "diffuse", version = env!("DIFFUSE_VERSION"), about = "Decentralized private LLM inference")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
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
    Host {
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value = "http://127.0.0.1:50051")]
        worker: String,
        #[arg(long, default_value = "0.0.0.0:9440")]
        listen: String,
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
        #[arg(long, default_value_t = 0.3)]
        overhead: f64,
        #[arg(long, default_value_t = true)]
        spawn_worker: bool,
        #[arg(long)]
        public_addr: Option<String>,
    },
    Query {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        image: Vec<String>,
        #[arg(long)]
        audio: Vec<String>,
        #[arg(long)]
        video: Vec<String>,
        #[arg(long)]
        media: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
        #[arg(long, default_value_t = 80)]
        max_tokens: usize,
        #[arg(long, default_value_t = 20)]
        steps: usize,
        #[arg(long, default_value_t = 4)]
        patches: usize,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    Models {
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
    },
    Chat {
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
        #[arg(long, default_value_t = false)]
        memory: bool,
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
    },
    Serve {
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
    },
}
