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
        /// Model to host. Omit to browse the marketplace and pick one.
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
    /// Query the network: discover a servable model, route through it, generate
    Query {
        #[arg(long)]
        model: String,
        #[arg(long)]
        prompt: String,
        /// Image to send with the prompt. Repeatable.
        #[arg(long)]
        image: Vec<String>,
        /// Audio clip to send with the prompt. Repeatable.
        #[arg(long)]
        audio: Vec<String>,
        /// Video to send with the prompt. Repeatable.
        #[arg(long)]
        video: Vec<String>,
        /// Any attachment; its kind is taken from the file extension.
        #[arg(long)]
        media: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        bootstrap: Vec<String>,
        #[arg(long, default_value_t = 80)]
        max_tokens: usize,
        /// Denoising steps, for a model that answers by diffusion.
        #[arg(long, default_value_t = 20)]
        steps: usize,
        /// Patches each denoising step is cut into across the nodes.
        #[arg(long, default_value_t = 4)]
        patches: usize,
        /// Seed, so the same prompt gives the same answer.
        #[arg(long, default_value_t = 0)]
        seed: u64,
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
        /// Longest answer per turn. For a model that answers with words this is
        /// a ceiling reached only if it never stops; for one that answers with
        /// audio or pixels it is the length of the answer itself.
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
    },
    /// Run a local OpenAI-compatible HTTP server in front of the network
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