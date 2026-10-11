"""Exercise publication against real Git repositories without network access."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.publication.publish import configure_ssh, git, publication_ref, publish


class PublicationRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "CI_PIPELINE_SOURCE": "push",
            "CI_COMMIT_REF_PROTECTED": "true",
            "CI_DEFAULT_BRANCH": "main",
            "CI_COMMIT_BRANCH": "main",
        }

    def test_protected_default_branch(self) -> None:
        self.assertEqual(publication_ref(self.environment), "refs/heads/main")

    def test_protected_version_tag(self) -> None:
        for tag in ("v1.2.3", "v0.0.0-demo.1"):
            with self.subTest(tag=tag):
                self.assertEqual(
                    publication_ref({**self.environment, "CI_COMMIT_TAG": tag}),
                    f"refs/tags/{tag}",
                )

    def test_other_pipeline_sources_refs_and_tags_are_rejected(self) -> None:
        cases = [
            {"CI_PIPELINE_SOURCE": "merge_request_event"},
            {"CI_PIPELINE_SOURCE": "web"},
            {"CI_COMMIT_REF_PROTECTED": "false"},
            {"CI_COMMIT_BRANCH": "feature/private-work"},
            {"CI_DEFAULT_BRANCH": "", "CI_COMMIT_BRANCH": ""},
            {"CI_COMMIT_TAG": "internal-checkpoint"},
            {"CI_COMMIT_TAG": "v1.2.3;echo unexpected"},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                publication_ref({**self.environment, **changes})

    def test_ssh_uses_file_variables_and_pinned_host_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            key = Path(temporary) / "key with spaces"
            hosts = Path(temporary) / "known_hosts"
            key.write_text("test key material")
            hosts.write_text("test host key")
            remote = configure_ssh(
                {
                    "GITHUB_REPOSITORY": "failed33/SlicerVesselRaySegmentation",
                    "GITHUB_SSH_PRIVATE_KEY": str(key),
                    "GITHUB_SSH_KNOWN_HOSTS": str(hosts),
                }
            )
            self.assertEqual(remote, "git@github.com:failed33/SlicerVesselRaySegmentation.git")
            self.assertEqual(key.stat().st_mode & 0o777, 0o600)
            command = os.environ["GIT_SSH_COMMAND"]
            self.assertIn("StrictHostKeyChecking=yes", command)
            self.assertIn("IdentitiesOnly=yes", command)
            self.assertNotIn("test key material", command)
            self.assertEqual(os.environ["GIT_TERMINAL_PROMPT"], "0")


class PublicationIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source = root / "gitlab"
        self.remote = root / "github.git"
        self.source.mkdir()
        git(self.source, "init", "-b", "main")
        git(self.source, "config", "user.name", "Publication test")
        git(self.source, "config", "user.email", "publication@example.invalid")
        git(self.source, "config", "commit.gpgsign", "false")
        git(self.source, "config", "tag.gpgsign", "false")
        git(root, "init", "--bare", "-b", "main", str(self.remote))
        self.commit = self.commit_demo("Demo revision 1\n")

    def commit_demo(self, content: str) -> str:
        (self.source / "demo.txt").write_text(content)
        git(self.source, "add", "demo.txt")
        git(self.source, "commit", "-m", "Update publication demo")
        commit = git(self.source, "rev-parse", "HEAD")
        git(self.source, "update-ref", "refs/remotes/origin/main", commit)
        return commit

    def publish_main(self) -> str:
        return publish(self.source, str(self.remote), "refs/heads/main", self.commit)

    def test_initial_push_preserves_commit_and_demo_bytes(self) -> None:
        self.assertEqual(self.publish_main(), self.commit)
        self.assertEqual(git(self.remote, "rev-parse", "main"), self.commit)
        self.assertEqual(git(self.remote, "show", "main:demo.txt"), "Demo revision 1")

    def test_feature_branch_and_unrelated_tag_are_excluded(self) -> None:
        git(self.source, "checkout", "-b", "feature/private-work")
        (self.source / "unpublished.txt").write_text("Unpublished demo content\n")
        git(self.source, "add", "unpublished.txt")
        git(self.source, "commit", "-m", "Unpublished demo branch")
        unpublished = git(self.source, "rev-parse", "HEAD")
        git(self.source, "tag", "internal-checkpoint")
        git(self.source, "checkout", "main")
        self.publish_main()
        refs = git(self.remote, "for-each-ref", "--format=%(refname)")
        self.assertEqual(refs, "refs/heads/main")
        with self.assertRaises(subprocess.CalledProcessError):
            git(self.remote, "cat-file", "-e", unpublished)

    def test_repeated_publication_is_idempotent(self) -> None:
        self.publish_main()
        self.assertEqual(self.publish_main(), self.commit)
        self.assertEqual(git(self.remote, "rev-list", "--count", "main"), "1")

    def test_update_and_fresh_clone_match_source(self) -> None:
        self.publish_main()
        self.commit = self.commit_demo("Demo revision 2\n")
        self.publish_main()
        clone = self.source.parent / "public-checkout"
        git(self.source.parent, "clone", str(self.remote), str(clone))
        self.assertEqual((clone / "demo.txt").read_text(), "Demo revision 2\n")
        self.assertEqual(git(clone, "rev-parse", "HEAD"), self.commit)
        self.assertEqual(git(clone, "rev-list", "--count", "main"), "2")

    def test_annotated_version_tag_retains_tag_object_and_message(self) -> None:
        self.publish_main()
        git(self.source, "tag", "-a", "v0.0.0-demo.1", "-m", "Publication demonstration")
        ref = "refs/tags/v0.0.0-demo.1"
        expected = git(self.source, "rev-parse", ref)
        self.assertEqual(publish(self.source, str(self.remote), ref, self.commit), expected)
        self.assertEqual(git(self.remote, "cat-file", "-t", ref), "tag")
        self.assertEqual(git(self.remote, "rev-parse", f"{ref}^{{commit}}"), self.commit)
        self.assertIn("Publication demonstration", git(self.remote, "cat-file", "-p", ref))

    def test_divergent_public_history_is_rejected(self) -> None:
        self.publish_main()
        git(self.source, "checkout", "--orphan", "unrelated")
        self.commit = self.commit_demo("Unrelated history\n")
        with self.assertRaises(subprocess.CalledProcessError):
            self.publish_main()
        self.assertEqual(git(self.remote, "show", "main:demo.txt"), "Demo revision 1")

    def test_existing_version_tag_cannot_be_rewritten(self) -> None:
        git(self.source, "tag", "-a", "v1.0.0", "-m", "First release")
        ref = "refs/tags/v1.0.0"
        original = publish(self.source, str(self.remote), ref, self.commit)
        self.commit = self.commit_demo("Another release\n")
        git(self.source, "tag", "-f", "-a", "v1.0.0", "-m", "Rewritten release")
        with self.assertRaises(subprocess.CalledProcessError):
            publish(self.source, str(self.remote), ref, self.commit)
        self.assertEqual(git(self.remote, "rev-parse", ref), original)

    def test_checkout_must_match_tested_commit(self) -> None:
        self.commit_demo("Changed checkout\n")
        with self.assertRaisesRegex(ValueError, "tested CI commit"):
            self.publish_main()
        self.assertEqual(git(self.remote, "for-each-ref"), "")

    def test_tag_must_match_tested_commit(self) -> None:
        git(self.source, "tag", "v1.0.0")
        self.commit = self.commit_demo("Changed checkout\n")
        with self.assertRaisesRegex(ValueError, "tested CI commit"):
            publish(self.source, str(self.remote), "refs/tags/v1.0.0", self.commit)
        self.assertEqual(git(self.remote, "for-each-ref"), "")

    def test_tag_on_unmerged_feature_branch_is_rejected(self) -> None:
        git(self.source, "checkout", "-b", "feature/unmerged")
        (self.source / "demo.txt").write_text("Unmerged demonstration\n")
        git(self.source, "commit", "-am", "Unmerged demonstration")
        commit = git(self.source, "rev-parse", "HEAD")
        git(self.source, "tag", "v1.0.0")
        with self.assertRaises(subprocess.CalledProcessError):
            publish(self.source, str(self.remote), "refs/tags/v1.0.0", commit)
        self.assertEqual(git(self.remote, "for-each-ref"), "")


if __name__ == "__main__":
    unittest.main()
