# Custom domain setup — emergentbiome.earth → GitHub Pages

Goal: serve the landing page and report at **https://emergentbiome.earth/** (joint venture
domain, managed by Alon on Cloudflare), while keeping `https://jaygut.github.io/MethaNet/`
fully intact as a fallback.

How it maps (this is a **project** Pages site, served from the `gh-pages` branch of
`jaygut/MethaNet`):

| Today | After custom domain |
| --- | --- |
| `jaygut.github.io/MethaNet/` | `https://emergentbiome.earth/` |
| `jaygut.github.io/MethaNet/report/` | `https://emergentbiome.earth/report/` |

GitHub serves the same `gh-pages` content at the **root** of the apex domain. Every asset on
the page uses relative paths, so it works at both URLs with no code change.

---

## 1. Cloudflare DNS records (give these to Alon)

In the Cloudflare dashboard for `emergentbiome.earth` → **DNS → Records**, add:

| Type | Name | Content (value) | Proxy status | TTL |
| --- | --- | --- | --- | --- |
| A | `@` | `185.199.108.153` | **DNS only** (grey cloud) | Auto |
| A | `@` | `185.199.109.153` | **DNS only** | Auto |
| A | `@` | `185.199.110.153` | **DNS only** | Auto |
| A | `@` | `185.199.111.153` | **DNS only** | Auto |
| AAAA | `@` | `2606:50c0:8000::153` | **DNS only** | Auto |
| AAAA | `@` | `2606:50c0:8001::153` | **DNS only** | Auto |
| AAAA | `@` | `2606:50c0:8002::153` | **DNS only** | Auto |
| AAAA | `@` | `2606:50c0:8003::153` | **DNS only** | Auto |
| CNAME | `www` | `jaygut.github.io` | **DNS only** | Auto |

(`@` is the apex/root `emergentbiome.earth`. These four A IPs and four AAAA IPs are GitHub's
official Pages addresses.)

### Critical Cloudflare gotcha — use "DNS only", not proxied
Set every record to **DNS only (grey cloud)**, not Proxied (orange cloud). With DNS-only,
GitHub provisions a real **Let's Encrypt** certificate for `emergentbiome.earth` automatically
and HTTPS just works. If the records are **proxied**, Let's Encrypt can't validate the domain,
GitHub falls back to a self-signed cert, and you then have to set Cloudflare SSL/TLS to **"Full"**
(never "Full (strict)") and add an ACME-challenge bypass rule — avoidable pain. Start DNS-only.

If you later want Cloudflare's CDN/WAF in front: only flip to proxied **after** the GitHub cert
is issued, set **SSL/TLS → Full**, and add a rule allowing `/.well-known/acme-challenge/*`
through (Cache: Bypass, Browser Integrity Check: Off) so the cert can renew.

Also: the domain must be **active on Cloudflare** (registrar pointing to Cloudflare's
nameservers) — Alon's onboarding covers this. If Cloudflare auto-adds a **CAA** record, make
sure it allows `letsencrypt.org`, or remove it; otherwise the cert won't issue.

---

## 2. Activate on GitHub (Jay) — do this AFTER DNS resolves

Order matters: **DNS first, then GitHub.** Confirm DNS is live:

```bash
dig emergentbiome.earth +short        # should return the four 185.199.10x.153 IPs
```

Then activate (one command — the `CNAME` file and canonical/share URLs are already staged):

```bash
cd web/emergentbiome-methanet
tools/publish_site.sh deploy --push
```

This publishes the site with a `CNAME` file containing `emergentbiome.earth`; GitHub
auto-detects it and sets the custom domain. Then:

1. Repo **Settings → Pages**: confirm "Custom domain" shows `emergentbiome.earth` and the DNS
   check passes.
2. Wait for "TLS certificate provisioned" (usually minutes, up to ~24h).
3. Tick **Enforce HTTPS**.
4. (Recommended for a JV) **Settings → Pages → Verify domain**: add the TXT record GitHub gives
   you in Cloudflare → prevents anyone else from claiming the domain on GitHub Pages.

Verify:
- `https://emergentbiome.earth/` and `https://emergentbiome.earth/report/` load.
- `https://jaygut.github.io/MethaNet/` 301-redirects to `https://emergentbiome.earth/`.

---

## 3. Your GitHub Pages are NOT deleted — guarantees

- Setting a custom domain only adds a `CNAME` file + a setting. The repo, the `gh-pages` branch,
  and all content are **untouched**.
- `jaygut.github.io/MethaNet/` keeps working — it **301-redirects** to the custom domain (same
  content, same branch). It is never removed.
- **Fully reversible:** to roll back, clear the custom domain in Settings → Pages, delete
  `web/emergentbiome-methanet/CNAME`, and run `tools/publish_site.sh deploy --push`. The
  `github.io` URL then serves directly again.
- The publish script now **preserves the `CNAME`** on every republish (it will never wipe the
  custom-domain binding), so future content updates won't break the domain.

---

## 4. One-line summary for Alon

> Add A records for `@` → `185.199.108.153`, `185.199.109.153`, `185.199.110.153`,
> `185.199.111.153` and AAAA records for `@` → `2606:50c0:8000::153`, `…8001::153`,
> `…8002::153`, `…8003::153`, plus a CNAME `www` → `jaygut.github.io`. **All set to "DNS only"
> (grey cloud), not proxied.** Once those resolve, I'll point GitHub Pages at
> `emergentbiome.earth` and enable HTTPS.
