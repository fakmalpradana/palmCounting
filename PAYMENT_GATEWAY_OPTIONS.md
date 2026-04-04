# Payment Gateway Options — Indonesia Individual Developer
**Palm Counter · Token Monetization Research**
*Prepared: 2026-04-04 · Scope: Individual/Perorangan (No PT/CV)*

---

## Context

- **Seller profile:** Individual developer (Fairuz Akmal Pradana), no registered company (no PT / no CV).
- **Use case:** Sell token credits in-app (one-time top-up). Payment via QRIS, GoPay, OVO, DANA, Virtual Account.
- **Stack:** Python / FastAPI backend on Google Cloud Run.
- **Key constraint:** No PT = no access to full business tiers on most gateways.

---

## Quick Verdict

| Gateway | Individual-Friendly | Best For |
|---|---|---|
| **Tripay** | ⭐⭐⭐⭐⭐ Best | First choice — KTP only, instant sandbox, cheapest VA |
| **Xendit** | ⭐⭐⭐⭐ Good | Premium API quality, needs KTP + bank account |
| **Duitku** | ⭐⭐⭐ OK | Lower e-wallet MDR, but NPWP required |

---

## 1. Xendit

### Registration Requirements (Individual)
| Requirement | Status |
|---|---|
| KTP | ✅ Required |
| NPWP | ⚠️ Optional — not required for individuals |
| Company documents (PT/CV/SIUP) | ❌ Not required for personal tier |
| Bank account | ✅ Required for withdrawal |

> Xendit explicitly supports personal (individual) accounts. No PT/CV needed.
> NPWP substitute accepted: KTP is sufficient for OVO, ShopeePay, Alfamart channels.
> Instant activation possible for low-risk individual accounts (limits applied until full KYC).

### Fee Schedule

| Payment Method | MDR / Fee |
|---|---|
| **QRIS** | **0.70%** per transaction (incl. PPN) |
| **OVO** | 2.73% (Indonesian Digital Content) |
| **DANA** | 1.50% (with PIN) / 3.00% (without PIN / recurring) |
| **ShopeePay** | ~2.00% |
| **Virtual Account** (BNI, BRI, Mandiri, Permata) | 0.5% + USD 0.20 ⚠️ |
| **Virtual Account BCA** | 0.5% + USD 0.20 ⚠️ |
| No monthly / setup fee | ✅ |
| Withdrawal fee | ✅ Free |

> ⚠️ VA fee is USD-based — at USD 1 = IDR 16.000, each VA transaction costs ~Rp 3,200 fixed.
> This is more expensive than Tripay for small token purchases (< Rp 50,000).

### API / Integration Quality
- **Docs:** `docs.xendit.co` — best documentation quality of all three gateways.
- **Python SDK:** Official `xendit-python` package on PyPI.
- **Webhooks:** HTTPS POST with signature verification (`x-callback-token` header).
- **Sandbox:** Full sandbox environment with test cards and mock e-wallets.
- **FastAPI fit:** Excellent. Clean REST API, predictable JSON responses, easy async integration.

### Verdict
Xendit has the best developer experience and most mature API. The individual tier is viable.
Main downside: VA fee is USD-denominated, which hurts for small IDR transactions.

---

## 2. Tripay

### Registration Requirements (Individual)
| Requirement | Status |
|---|---|
| KTP | ✅ Required |
| NPWP | ✅ Not required — KTP is sufficient |
| Company documents (PT/CV) | ❌ Not required |
| Bank account | ✅ Required for withdrawal |

> **Best gateway for individuals in Indonesia.** Designed with solo developers and small sellers in mind.
> Fastest registration: KTP only, approved typically within 1 business day.
> No NPWP needed — unique advantage over Duitku.

### Fee Schedule

| Payment Method | MDR / Fee |
|---|---|
| **QRIS** | **Rp 750 flat + 0.70%** per transaction |
| **OVO** | 3.00% (min Rp 1,000) |
| **ShopeePay** | 3.00% (min Rp 1,000) |
| **GoPay** | ~3.00% (same tier as OVO) |
| **DANA** | ~3.00% |
| **VA BRI / BNI / Permata / Mandiri** | **Rp 4,250 flat** |
| **VA BCA** | **Rp 5,500 flat** |
| No monthly / setup fee | ✅ |

> QRIS flat-fee approach (Rp 750 + 0.7%) is slightly more expensive than 0.7% alone for large amounts,
> but very predictable for token top-ups (e.g., Rp 100K → fee = Rp 750 + Rp 700 = Rp 1,450).

### API / Integration Quality
- **Docs:** `tripay.co.id/developer` — well-structured, clean REST API.
- **Python examples:** Provided in callback/webhook docs.
- **Webhooks:** HMAC-SHA256 signature validation, retry up to 3× at 2-min intervals, IP whitelisting available.
- **Sandbox:** Full sandbox at `tripay.co.id/api-sandbox/` with built-in Callback Tester tool.
- **FastAPI fit:** Very good. Simple auth (`Authorization: Bearer {api_key}`), JSON in/out, consistent error format.

### Verdict
**Recommended first choice for Palm Counter.** Easiest onboarding (KTP only), solid API, good sandbox,
and the most complete payment channels accessible to individuals without NPWP.

---

## 3. Duitku

### Registration Requirements (Individual)
| Requirement | Status |
|---|---|
| KTP | ✅ Required |
| NPWP | ⚠️ Required (or Student Card as substitute) |
| Company documents | ❌ Not required |
| Bank account | ✅ Required |

> NPWP or equivalent is required — this is the main friction point compared to Tripay.
> BCA Virtual Account **not available for personal merchants** (workaround: QRIS instead of BCA VA).

### Fee Schedule

| Payment Method | MDR / Fee |
|---|---|
| **QRIS** | **0.70%** (no flat fee — cheapest for large transactions) |
| **OVO** | 1.67% |
| **DANA** | 1.67% |
| **LinkAja** | 1.67% (or Rp 3,330 flat) |
| **ShopeePay** | 2.00% |
| **OVO Digital** | 3.03% |
| **ShopeePay Digital** | 4.00% |
| **VA Mandiri** | **Rp 4,000 flat** |
| **VA BCA** | **Rp 5,000 flat** ⚠️ Not available for personal |
| **VA BRI / BNI / others** | **Rp 3,000 flat** (cheapest VA) |
| No monthly / setup fee | ✅ |
| Settlement | D+2 for QRIS |

> Duitku has the **lowest e-wallet MDR** (1.67% for OVO/DANA vs 2.73–3% on Xendit/Tripay).
> Also cheapest VA flat fee (Rp 3,000 for non-BCA).

### API / Integration Quality
- **Docs:** `docs.duitku.com` — comprehensive, supports SNAP API standard (Bank Indonesia standard).
- **Python:** No official SDK; use `requests` or `httpx` directly with their REST API.
- **Webhooks:** HTTP POST callback to merchant URL, HMAC-SHA256 signature.
- **Sandbox:** Available via Duitku dashboard (requires account registration).
- **FastAPI fit:** Good. SNAP API is standardized and well-documented. Slightly more setup than Tripay.

### Verdict
Best fee rates for e-wallets. But NPWP requirement is a barrier if you don't have one yet.
Worth using once you have NPWP. For now, use Tripay first, migrate to Duitku later if cost matters.

---

## Side-by-Side Comparison

| | **Xendit** | **Tripay** | **Duitku** |
|---|---|---|---|
| **Individual (no PT)** | ✅ Yes | ✅ Yes | ✅ Yes |
| **KTP only** | ✅ Yes | ✅ Yes | ⚠️ Needs NPWP too |
| **NPWP required** | ❌ No | ❌ No | ⚠️ Yes |
| **QRIS MDR** | 0.70% | Rp 750 + 0.70% | 0.70% |
| **GoPay fee** | ~2.73% | ~3.00% | ~1.67% |
| **OVO fee** | 2.73% | 3.00% | 1.67% |
| **VA (non-BCA) fee** | 0.5% + ~Rp 3,200 | Rp 4,250 flat | Rp 3,000 flat |
| **Python SDK** | ✅ Official | ❌ DIY | ❌ DIY |
| **Sandbox** | ✅ Full | ✅ Full | ✅ With account |
| **Webhook quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Doc quality** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **FastAPI ease** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Monthly/setup fee** | None | None | None |

---

## Recommendation for Palm Counter

### Phase 1 (Now — No NPWP ready): **Tripay**
- Register with KTP only — done in ~1 day.
- Enables QRIS, OVO, ShopeePay, VA (BRI/BNI/BCA) for token purchases.
- Solid API, Python examples in docs, good sandbox.
- Token top-up at Rp 50K: fee = ~Rp 1,100 (QRIS) or Rp 4,250 (VA).

### Phase 2 (Later — NPWP obtained): **Duitku**
- Lower e-wallet MDR (1.67% vs 3%) significantly reduces cost for high-volume users.
- Migrate only if e-wallet usage becomes dominant.

### If needing enterprise-grade SDK/support: **Xendit**
- Best for scaling, official Python SDK, most mature platform.
- Consider when monthly volume justifies their support tier.

---

## Token Top-Up Fee Examples (Rp 100,000 purchase)

| Method | Tripay | Xendit | Duitku |
|---|---|---|---|
| QRIS | Rp 1,450 | Rp 700 | Rp 700 |
| OVO | Rp 3,000 | Rp 2,730 | Rp 1,670 |
| VA (non-BCA) | Rp 4,250 | ~Rp 3,700 | Rp 3,000 |

---

*Sources: Xendit Help Center, Xendit Pricing, Tripay Developer Docs, Duitku FAQ & Pricing pages.*
