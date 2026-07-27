use clap::Parser;
use diffuse_daemon::commands;
use diffuse_daemon::config::{Cli, Command};
use diffuse_daemon::identity::Identity;
use diffuse_daemon::serve;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let cli = Cli::parse();
    let identity = Identity::generate();

    match cli.command {
        Command::Plan {
            model,
            worker,
            overhead,
            bootstrap,
        } => commands::plan(&model, &worker, overhead, &bootstrap, &identity).await,
        Command::Demo {
            stage_a,
            stage_b,
            model,
            prompt,
            spares,
        } => commands::demo(stage_a, stage_b, model, prompt, spares, identity).await,
        Command::Host {
            model, worker, listen, bootstrap, overhead, spawn_worker, public_addr,
        } => commands::host(model.as_deref(), &worker, &listen, &bootstrap, overhead, spawn_worker, public_addr, identity).await,
        Command::Query {
            model,
            prompt,
            image,
            audio,
            video,
            media,
            bootstrap,
            max_tokens,
            steps,
            patches,
            seed,
        } => {
            let attachments = commands::collect_attachments(&image, &audio, &video, &media)?;
            commands::query(
                &model, &prompt, attachments, &bootstrap, max_tokens, steps, patches, seed,
                identity,
            )
            .await
        }
        Command::Models { bootstrap } => commands::models(&bootstrap, identity).await,
        Command::Chat {
            bootstrap,
            memory,
            max_tokens,
        } => commands::chat(&bootstrap, memory, max_tokens, identity).await,
        Command::Serve {
            port,
            host,
            model,
            bootstrap,
        } => serve::serve(&host, port, model, &bootstrap, identity).await,
    }
}
