# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it
responsibly by opening a private security advisory on GitHub or contacting
the maintainers directly.

**Do NOT open a public issue for security vulnerabilities.**

## Scope

This project is a framework for strategy development and does **not** store
or transmit financial credentials by design. However, users configure their
own API keys via environment variables (`.env`).

## Best Practices for Users

- Never commit `.env` files or API keys to version control
- Use GitHub Secrets or similar services for CI/CD environments
- Review `.env.example` for the expected configuration format
- Run `gitleaks` or similar tools before pushing to public repositories

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
