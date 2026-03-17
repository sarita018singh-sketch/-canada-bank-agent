"""
Canadian Savings Account Competitor Analysis Agent
---------------------------------------------------
FastAPI backend with an OpenAI-powered comparison agent.
Reads structured data from nested JSON format (banks → accounts).

Run with: uvicorn app:app --reload --port 8000
Then open: http://localhost:8000
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from openai import OpenAI
from pydantic import BaseModel

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────

app = FastAPI(
    title="Canadian Savings Account Competitor Analysis",
    description="AI-powered comparison tool for Canadian bank savings accounts",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = Path(__file__).parent / "bank_data.json"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# ──────────────────────────────────────────────
# Rate Limiting (protects your OpenAI bill)
# ──────────────────────────────────────────────
DAILY_LIMIT = int(os.environ.get("DAILY_QUERY_LIMIT", "10"))
_usage: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(ip: str) -> tuple[bool, int]:
    """Check if an IP has queries remaining today."""
    now = time.time()
    day_ago = now - 86400
    _usage[ip] = [t for t in _usage[ip] if t > day_ago]
    remaining = DAILY_LIMIT - len(_usage[ip])
    if remaining <= 0:
        return False, 0
    return True, remaining


def record_usage(ip: str):
    _usage[ip].append(time.time())


# ──────────────────────────────────────────────
# Data Loading — handles new nested JSON format
# ──────────────────────────────────────────────

def load_raw_data() -> dict:
    """Load the raw nested JSON data."""
    if not DATA_FILE.exists():
        return {"metadata": {}, "banks": []}
    with open(DATA_FILE) as f:
        return json.load(f)


def flatten_products(raw: dict) -> list[dict]:
    """Flatten the nested banks→accounts structure into a flat product list.

    This converts the new nested format into the flat format the frontend expects.
    """
    products = []
    for bank in raw.get("banks", []):
        bank_name = bank.get("bank_name", "")
        full_name = bank.get("full_name", "")
        bank_type = bank.get("bank_type", "")

        for acct in bank.get("accounts", []):
            # Build promotional rate string
            promo_rate = acct.get("promo_rate_pct")
            promo_days = acct.get("promo_duration_days")
            promo_expiry = acct.get("promo_expiry")
            if promo_rate and promo_rate > 0:
                if promo_days:
                    promo_str = f"{promo_rate}% for {promo_days} days"
                elif promo_expiry:
                    promo_str = f"{promo_rate}% (until {promo_expiry})"
                else:
                    promo_str = f"{promo_rate}%"
                promo_duration = f"{promo_days} days" if promo_days else (f"Until {promo_expiry}" if promo_expiry else "Ongoing")
            else:
                promo_str = "N/A"
                promo_duration = "N/A"

            # Build rate tiers string
            tiers = acct.get("rate_tiers", [])
            tier_parts = []
            for t in tiers:
                if "rate_pct" in t and "min_balance" in t:
                    max_b = t.get("max_balance")
                    if max_b:
                        tier_parts.append(f"${t['min_balance']:,.0f}-${max_b:,.0f}: {t['rate_pct']}%")
                    else:
                        tier_parts.append(f"${t['min_balance']:,.0f}+: {t['rate_pct']}%")
                elif "bonus_pct" in t:
                    tier_parts.append(f"+{t['bonus_pct']}% bonus ({t.get('condition', 'conditional')})")
                elif "tier" in t:
                    tier_parts.append(f"{t['tier']}: {t.get('rate_pct', '?')}%")
            rate_tiers_str = " | ".join(tier_parts) if tier_parts else "Flat rate"

            # Build features string
            features = acct.get("features", [])
            features_str = ", ".join(features) if features else "N/A"

            # Build new features string
            new_feats = acct.get("new_features", [])
            new_feats_str = ", ".join(new_feats) if new_feats else "None"

            # Source info
            source_info = acct.get("source_freshness", {})
            source = source_info.get("source_name", "Unknown")

            # Min balance and fee formatting
            min_bal = acct.get("min_balance", 0)
            min_bal_str = f"${min_bal:,.0f}" if min_bal else "$0"
            fee = acct.get("monthly_fee", 0)
            fee_str = f"${fee:,.0f}" if fee else "$0"

            product = {
                "bank_name": bank_name,
                "bank_full_name": full_name,
                "bank_type": bank_type,
                "account_name": acct.get("account_name", ""),
                "base_rate_pct": f"{acct.get('base_rate_pct', 0):.2f}%",
                "base_rate_num": float(acct.get("base_rate_pct", 0)),
                "promotional_rate": promo_str,
                "promo_rate_num": float(promo_rate) if promo_rate else 0,
                "promo_duration": promo_duration,
                "promo_details": acct.get("promo_details"),
                "promo_expiry": promo_expiry,
                "minimum_balance": min_bal_str,
                "monthly_fee": fee_str,
                "cdic_insured": acct.get("cdic_insured", False),
                "cdic_partner": acct.get("cdic_partner"),
                "e_transfer": acct.get("e_transfer", False),
                "e_transfer_details": acct.get("e_transfer_notes"),
                "joint_account": acct.get("joint_account", False),
                "joint_account_details": acct.get("joint_account_notes"),
                "rate_tiers": rate_tiers_str,
                "key_features": features_str,
                "new_features": new_feats_str,
                "base_rate_note": acct.get("base_rate_note"),
                "notice_period_days": acct.get("notice_period_days"),
                "source": source,
                "last_updated": acct.get("last_updated", ""),
                "verification": json.dumps(acct["verification"]) if isinstance(acct.get("verification"), dict) else str(acct.get("verification", "")),
            }
            products.append(product)

    return products


def load_data() -> dict:
    """Load data and return in flat format for API endpoints."""
    raw = load_raw_data()
    products = flatten_products(raw)
    metadata = raw.get("metadata", {})
    # Build compatible metadata
    return {
        "metadata": {
            "total_banks": metadata.get("total_banks", 0),
            "total_products": metadata.get("total_products", len(products)),
            "last_full_refresh": metadata.get("date_generated", "unknown"),
            "disclaimer": metadata.get("disclaimer", ""),
            "disclaimers": [metadata.get("disclaimer", "")] if metadata.get("disclaimer") else [],
            "sources": metadata.get("sources", []),
            "bank_of_canada_rate": metadata.get("bank_of_canada_rate", {}),
            "changelog": metadata.get("changelog", []),
            "expiry_alerts": metadata.get("expiry_alerts", []),
            "validation_flags": metadata.get("validation_flags", []),
        },
        "products": products,
    }


def get_products_by_bank(name: str) -> list[dict]:
    """Find all products for a bank (case-insensitive partial match)."""
    data = load_data()
    name_lower = name.lower().strip()
    return [p for p in data["products"] if name_lower in p["bank_name"].lower()]


def get_unique_banks() -> list[dict]:
    """Get unique bank names with their types."""
    data = load_data()
    seen = {}
    for p in data["products"]:
        if p["bank_name"] not in seen:
            seen[p["bank_name"]] = {
                "bank_name": p["bank_name"],
                "bank_full_name": p["bank_full_name"],
                "bank_type": p["bank_type"],
                "product_count": 0,
            }
        seen[p["bank_name"]]["product_count"] += 1
    return list(seen.values())


def build_full_context() -> str:
    """Build the complete data context string for the LLM agent."""
    data = load_data()
    lines = []

    for p in data["products"]:
        promo = f" | PROMO: {p['promotional_rate']} ({p['promo_duration']})" if p["promotional_rate"] != "N/A" else ""
        expiry = f" | EXPIRES: {p['promo_expiry']}" if p.get("promo_expiry") else ""
        base_note = f" [{p['base_rate_note']}]" if p.get("base_rate_note") else ""
        notice = f" | NOTICE PERIOD: {p['notice_period_days']} days" if p.get("notice_period_days") else ""

        lines.append(
            f"--- {p['bank_name']} - {p['account_name']} ---\n"
            f"  Bank Type: {p['bank_type']}\n"
            f"  Base Rate: {p['base_rate_pct']}{base_note}{promo}{expiry}\n"
            f"  Promo Details: {p.get('promo_details', 'N/A')}\n"
            f"  Min Balance: {p['minimum_balance']} | Fee: {p['monthly_fee']}{notice}\n"
            f"  CDIC: {'Yes' if p['cdic_insured'] else 'No'}"
            f"{' (via ' + p['cdic_partner'] + ')' if p.get('cdic_partner') else ''} | "
            f"e-Transfer: {p.get('e_transfer_details') or ('Yes' if p['e_transfer'] else 'No')} | "
            f"Joint: {'Yes' if p['joint_account'] else 'No'}"
            f"{' (' + p['joint_account_details'] + ')' if p.get('joint_account_details') else ''}\n"
            f"  Rate Tiers: {p['rate_tiers']}\n"
            f"  Key Features: {p['key_features']}\n"
            f"  Recent Changes: {p.get('new_features', 'None')}\n"
            f"  Updated: {p['last_updated']} | Source: {p['source']}"
        )

    raw = load_raw_data()
    meta = raw.get("metadata", {})

    # Add Bank of Canada rate context
    boc = meta.get("bank_of_canada_rate", {})
    boc_context = ""
    if boc:
        boc_context = (
            f"\n\nBANK OF CANADA POLICY RATE: {boc.get('rate_pct', '?')}%\n"
            f"Last Decision: {boc.get('last_decision_date', '?')} | "
            f"Next Decision: {boc.get('next_decision_date', '?')} | "
            f"Direction: {boc.get('direction', '?')}\n"
            f"Notes: {boc.get('notes', '')}"
        )

    # Add expiry alerts
    alerts = meta.get("expiry_alerts", [])
    alert_context = ""
    if alerts:
        alert_lines = [f"  - {a['bank']} {a['account']}: {a.get('promo_rate_pct', '?')}% expires {a.get('promo_expiry', a.get('expiry_date', '?'))} ({a.get('alert', 'unknown')})" for a in alerts]
        alert_context = "\n\nEXPIRY ALERTS:\n" + "\n".join(alert_lines)

    disclaimers = meta.get("disclaimer", "Rates subject to change. Not financial advice.")

    return (
        f"DATABASE: {len(data['products'])} savings products across "
        f"{meta.get('total_banks', '?')} Canadian banks\n"
        f"Data as of: {meta.get('date_generated', 'unknown')}"
        f"{boc_context}"
        f"{alert_context}\n\n"
        + "\n\n".join(lines)
        + f"\n\nDISCLAIMER: {disclaimers}"
    )


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────

class CompareRequest(BaseModel):
    query: str


class CompareResponse(BaseModel):
    query: str
    response: str
    banks_referenced: list[str]
    products_referenced: list[str]
    timestamp: str


# ──────────────────────────────────────────────
# OpenAI Agent
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior competitive intelligence analyst specializing in Canadian retail banking savings products.

You work for an internal bank strategy team. Your job is to provide precise, data-driven comparisons and analysis of savings account products across Canadian financial institutions.

You have access to a curated database of savings products across 10 Canadian banks (Big 5, digital banks, and fintechs).

ANALYSIS RULES:
1. ALWAYS cite exact numbers from the database. Never approximate or guess rates.
2. Distinguish clearly between BASE rates (what you actually get long-term) and PROMOTIONAL rates (temporary, usually for new customers only).
3. When comparing banks, structure your analysis by: Rate Comparison, Fee Structure, Features (CDIC, e-Transfer, joint accounts), Promo Analysis, and Strategic Assessment.
4. Note that banks often have MULTIPLE savings products — always specify which product you're discussing.
5. Flag promotional rates that are expiring soon (within the next month).
6. When asked "which is best", always clarify: best for what? (highest base rate, best promo, most features, best for large balances, etc.)
7. Note important caveats like: notice periods (EQ Bank), tiered rates that require high balances (TD Growth, Wealthsimple), or restrictions on joint accounts (Neo, Wealthsimple).
8. Always mention the data source date so analysts know freshness.
9. Be direct and analytical. No fluff. These are banking professionals.
10. End every comparison with a "Competitive Positioning" note — what does this mean for our bank's strategy?

IMPORTANT PRODUCT NUANCES TO REMEMBER:
- Neo Financial restructured in Dec 2025: old 4% HISA is gone. New Neo Savings offers up to 3% (tiered, need $20K+ for max rate). Legacy HISA dropped to 1.25%.
- EQ Bank Notice Savings accounts require 10-day or 30-day withdrawal notice — higher rate but restricted liquidity.
- EQ Bank Personal Account base rate dropped from 1.25% to 1.00% (following BoC cuts). With $2K+/month direct deposit, gets 2.75%.
- Wealthsimple rates are tier-dependent: Core (1.25%), Premium (1.75%), Generation (2.25%) — most customers get Core. +0.50% direct deposit bonus for Core/Premium.
- CIBC eAdvantage has the highest promo rate (5.25%) but only for 3 months. Smart Interest feature adds 0.25%.
- Many Big 5 "High Interest Savings" accounts actually offer near-zero rates (0.01%).
- Scotiabank MomentumPLUS has a complex premium period structure (90-360 days) with package boosts.
- TD Growth Savings is tiered: only 1.50% at $500K+, starts at 0.01% for under $10K.
- Check the EXPIRY ALERTS section for promos ending soon.
- Bank of Canada rate context is provided — use it when discussing rate trends.

TODAY'S DATE: {today}

COMPLETE DATABASE:
{bank_data}"""


def get_agent_response(query: str) -> tuple[str, list[str], list[str]]:
    """Get a response from the OpenAI comparison agent."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured. Set it as an environment variable.",
        )

    client = OpenAI(api_key=api_key)
    bank_data_context = build_full_context()
    today = datetime.now().strftime("%Y-%m-%d")

    system_message = SYSTEM_PROMPT.format(today=today, bank_data=bank_data_context)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=3000,
    )

    answer = response.choices[0].message.content
    answer_lower = answer.lower()

    # Detect referenced banks and products
    data = load_data()
    banks_referenced = list({
        p["bank_name"] for p in data["products"]
        if p["bank_name"].lower() in answer_lower
    })
    products_referenced = list({
        f"{p['bank_name']} {p['account_name']}" for p in data["products"]
        if p["account_name"].lower() in answer_lower
    })

    return answer, banks_referenced, products_referenced


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main dashboard."""
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(html_path)


@app.get("/api/products")
async def list_all_products():
    """Return all savings products with full details."""
    data = load_data()
    return {
        "metadata": data["metadata"],
        "products": data["products"],
        "count": len(data["products"]),
    }


@app.get("/api/banks")
async def list_banks():
    """Return unique bank list with product counts."""
    banks = get_unique_banks()
    return {"banks": banks, "count": len(banks)}


@app.get("/api/banks/{bank_name}")
async def get_bank_products(bank_name: str):
    """Return all savings products for a specific bank."""
    products = get_products_by_bank(bank_name)
    if not products:
        raise HTTPException(status_code=404, detail=f"Bank '{bank_name}' not found")
    return {
        "bank_name": products[0]["bank_full_name"],
        "bank_type": products[0]["bank_type"],
        "products": products,
        "product_count": len(products),
    }


@app.get("/api/rates/top")
async def top_rates(limit: int = 10, include_promos: bool = False):
    """Return top savings rates. By default base rates only."""
    data = load_data()
    if include_promos:
        sorted_products = sorted(
            data["products"],
            key=lambda p: max(p["base_rate_num"], p.get("promo_rate_num", 0)),
            reverse=True,
        )
    else:
        sorted_products = sorted(data["products"], key=lambda p: p["base_rate_num"], reverse=True)
    return {
        "top_rates": sorted_products[:limit],
        "include_promos": include_promos,
        "as_of": data["metadata"].get("last_full_refresh", "unknown"),
    }


@app.post("/api/compare", response_model=CompareResponse)
async def agent_compare(request: CompareRequest, req: Request):
    """AI agent-powered analysis of savings accounts."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Rate limiting
    ip = req.client.host if req.client else "unknown"
    allowed, remaining = check_rate_limit(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="You've used all 10 free queries for today. Come back tomorrow!",
        )

    answer, banks_ref, products_ref = get_agent_response(request.query)
    record_usage(ip)

    return CompareResponse(
        query=request.query,
        response=answer,
        banks_referenced=banks_ref,
        products_referenced=products_ref,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/remaining")
async def remaining_queries(req: Request):
    """Check how many queries a visitor has left today."""
    ip = req.client.host if req.client else "unknown"
    _, remaining = check_rate_limit(ip)
    return {"remaining": remaining, "daily_limit": DAILY_LIMIT}


@app.get("/api/health")
async def health_check():
    """Health check."""
    data = load_data()
    api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "status": "healthy" if api_key_set else "degraded",
        "api_key_configured": api_key_set,
        "products_in_database": len(data.get("products", [])),
        "banks_tracked": data.get("metadata", {}).get("total_banks", 0),
        "last_refresh": data.get("metadata", {}).get("last_full_refresh", "never"),
        "timestamp": datetime.now().isoformat(),
    }
