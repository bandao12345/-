import unittest
from flask import Flask, jsonify
from demo_access import install_demo_access


class DemoAccessTests(unittest.TestCase):
    def make_app(self, token="t" * 32):
        app = Flask(__name__)
        app.config["DEMO_ACCESS_TOKEN"] = token
        app.add_url_rule("/health", endpoint="health", view_func=lambda: jsonify(ready=True))
        app.add_url_rule("/", endpoint="index", view_func=lambda: "demo")
        app.add_url_rule("/api/write", endpoint="write", methods=["POST"], view_func=lambda: jsonify(success=True))
        install_demo_access(app)
        return app

    def test_unauthorized_requests_do_not_reveal_workspace(self):
        client = self.make_app().test_client()
        self.assertEqual(client.get("/").status_code, 401)
        self.assertEqual(client.post("/api/write").status_code, 401)
        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.get("/demo-login?token=wrong").status_code, 401)

    def test_login_redirect_hides_token_and_sets_httponly_cookie(self):
        client = self.make_app().test_client()
        response = client.get("/demo-login?token=" + "t" * 32,
                              headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["Location"], "/")
        self.assertIn("HttpOnly", response.headers["Set-Cookie"])
        self.assertIn("Secure", response.headers["Set-Cookie"])
        self.assertEqual(client.get("/").status_code, 200)

    def test_demo_mutation_limit(self):
        client = self.make_app().test_client()
        client.get("/demo-login?token=" + "t" * 32)
        for _ in range(20):
            self.assertEqual(client.post("/api/write").status_code, 200)
        self.assertEqual(client.post("/api/write").status_code, 429)

    def test_default_local_workspace_needs_no_demo_token(self):
        client = self.make_app("").test_client()
        self.assertEqual(client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
