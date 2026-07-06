# Contributing to Diffuse

Contributions are welcome: code, review, docs, or careful bug reports.

## Ground rules

- Be honest about what code does. Accuracy over optimism, especially on
  security and privacy claims.
- No hardcoded model-specific values. Diffuse aims to work with any model.
- Small, focused pull requests over large ones.

## Build and test

```
cargo build && cargo test            # Rust daemon
cd worker && python -m pytest        # Python worker
cd worker && ./scripts/gen_proto.sh  # after editing proto/
```

Make sure `cargo build` and `cargo test` pass before opening a PR.

## Security issues

Do not use public issues for vulnerabilities. See [`SECURITY.md`](SECURITY.md).

## License

Contributions are licensed under AGPL-3.0-or-later.