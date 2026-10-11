# SlicerVesselRaySegmentation

This repository currently demonstrates the publication infrastructure for a planned
3D Slicer extension. It does not yet contain the extension implementation.

The planned workflow is ARG inference, disconnected-component removal, centerline
generation and manual editing, then ModelO inference.

## Development and publication

GitLab is the authoritative development repository. Its CI pipeline runs the
publication tests before pushing the tested default-branch commit or a protected
version tag to GitHub. Feature branches stay in GitLab. GitHub receives ordinary
Git commits with the same IDs, and published history is never force-pushed.

Run the tests with Python 3.12 and Git installed:

```sh
python3 -m unittest discover -s tests -v
```

The tests create temporary local Git repositories. They need no credentials or
network access.

## GitLab setup

Set `main` as the protected default branch and protect version tags matching `v*`.
Configure these protected file variables with environment scope `public`:

- `GITHUB_SSH_PRIVATE_KEY`: a dedicated SSH deploy key with write access to this
  GitHub repository only.
- `GITHUB_SSH_KNOWN_HOSTS`: GitHub's SSH host public keys, obtained from its
  authenticated HTTPS metadata API.

The pipeline uses these files directly without printing their contents. Publication
requires a push pipeline on a protected ref. Merge request, feature-branch and web
pipelines run tests only. Version tags must point to commits on the GitLab `main`
history. Create an annotated release tag such as `v1.0.0` in GitLab after the
corresponding default-branch pipeline succeeds.

This pipeline publishes the dedicated extension repository's history. Keep every
file and commit in this repository suitable for public release. Research sources,
private deployment settings, model weights and experimental data belong outside
this repository.

## License

Apache License 2.0. See [LICENSE.txt](LICENSE.txt).
