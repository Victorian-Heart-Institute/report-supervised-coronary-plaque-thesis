# Privacy and safe contributions

Keep this repository free of clinical images, reports, private manifests,
per-patient predictions, annotations, credentials, private keys and institutional
filesystem locations. Use synthetic study codes for examples. Pretrained weights
and private data belong outside Git; do not upload them even to a private branch.

The source and aggregate-result checksums are deliberate reproducibility metadata,
not credentials. Generic example paths and public upstream model URLs are also
deliberate. No institution-specific storage connection or access configuration is
needed in the distributed code.

Use your GitHub no-reply commit address. Enable the local staged-file guard:

```bash
git config core.hooksPath .githooks
python3 scripts/privacy_audit.py --all-history
```

When installed, the hook also runs Gitleaks. Before publication, separately run:

```bash
gitleaks git --log-opts=--all --redact --ignore-gitleaks-allow .
```

These checks complement human review; they cannot certify that every possible
sensitive fact has been detected. Do not post suspected secrets in issues or
scanner logs. Revoke exposed credentials first, then remove affected content and
history. Rewriting branches/tags does not necessarily delete GitHub caches or
other people's clones; contact GitHub Support if server-side removal is needed.
