"""Generate a realistic TwitterSupport synthetic corpus when Kaggle is unavailable.

Used only as fallback so the pipeline remains runnable. Prefer real Kaggle data.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from src.data.prepare import attach_weak_labels, save_processed
from src.utils.paths import ensure_dirs

TEMPLATES = {
    "account_access": [
        "@TwitterSupport I can't log into my account after resetting my password",
        "@TwitterSupport password reset email never arrives please help",
        "@TwitterSupport locked out of my account 2FA not working",
        "@TwitterSupport cannot sign in on the app, keeps saying wrong password",
    ],
    "account_compromised": [
        "@TwitterSupport my account was hacked and they changed the email",
        "@TwitterSupport someone took over my account posting spam",
        "@TwitterSupport unauthorized tweets from my handle please help",
        "@TwitterSupport account compromised overnight, password changed",
    ],
    "suspension_or_lock": [
        "@TwitterSupport why was my account suspended with no explanation?",
        "@TwitterSupport account locked for weeks, appeal ignored",
        "@TwitterSupport banned after a false report, need review",
        "@TwitterSupport permanent suspension seems wrong please escalate",
    ],
    "verification": [
        "@TwitterSupport how do I get verified?",
        "@TwitterSupport verification application pending for months",
        "@TwitterSupport blue check disappeared from my account",
    ],
    "app_or_bug": [
        "@TwitterSupport the iOS app keeps crashing when I open DMs",
        "@TwitterSupport notifications stopped working after the update",
        "@TwitterSupport timeline not loading on Android",
        "@TwitterSupport app freezes every time I upload a video",
    ],
    "ads_or_promoted": [
        "@TwitterSupport my promoted tweet was rejected without reason",
        "@TwitterSupport charged twice for an ad campaign",
        "@TwitterSupport Ads Manager won't let me edit my campaign",
    ],
    "safety_or_abuse": [
        "@TwitterSupport this account is threatening me and reports do nothing",
        "@TwitterSupport getting harassed daily please help",
        "@TwitterSupport abusive DMs and hate replies, need action",
    ],
    "privacy_or_data": [
        "@TwitterSupport how do I download all my Twitter data?",
        "@TwitterSupport my email seems exposed somehow",
        "@TwitterSupport privacy settings reset themselves",
    ],
    "feature_how_to": [
        "@TwitterSupport how do I mute words on Twitter?",
        "@TwitterSupport how can I pin a tweet to my profile?",
        "@TwitterSupport how to create a list and add people?",
    ],
    "spam_or_bots": [
        "@TwitterSupport my mentions are full of spam bots",
        "@TwitterSupport scam account impersonating brands",
        "@TwitterSupport phishing links flooding my replies",
    ],
    "thanks_or_other": [
        "@TwitterSupport thanks for the quick help!",
        "@TwitterSupport appreciate the follow-up",
        "@TwitterSupport hello??",
    ],
}

REPLIES = {
    "account_access": [
        "Hi there. Sorry you're having trouble signing in. Can you try resetting your password from a browser and reply with what error you see? ^TS",
        "Thanks for reaching out. Please DM us your username so we can look into the login issue. ^TS",
    ],
    "account_compromised": [
        "We're sorry to hear this. Please secure your account via the hacked flow at https://help.twitter.com and DM us your username. ^TS",
        "Thanks for flagging. Follow us and DM your @handle so we can investigate the unauthorized access. ^TS",
    ],
    "suspension_or_lock": [
        "We know how important this is. Please use the appeal form linked in the suspension notice and DM us your username. ^TS",
        "Thanks for contacting us. Share your @handle via DM so the right team can review the account status. ^TS",
    ],
    "verification": [
        "Thanks for asking. Verification details are here: https://help.twitter.com — we can't share timelines publicly. ^TS",
        "Hi! Please review the verification criteria on our Help Center. DM us if you have a specific error. ^TS",
    ],
    "app_or_bug": [
        "Sorry for the trouble. Which OS/app version are you on? Try reinstalling and let us know if it persists. ^TS",
        "Thanks for the report. Can you send a screenshot via DM and confirm device + app version? ^TS",
    ],
    "ads_or_promoted": [
        "Sorry about the ads issue. Please DM your Ads account ID so we can route this to the Ads team. ^TS",
        "Thanks for reaching out. Check Ads Manager billing first; if it still looks wrong, DM us details. ^TS",
    ],
    "safety_or_abuse": [
        "We're sorry you're dealing with this. Please report the account(s) and DM us links so Safety can review. ^TS",
        "Thanks for letting us know. Use the report flow, then DM us the usernames involved. ^TS",
    ],
    "privacy_or_data": [
        "You can request your data archive in Settings. If something looks wrong, DM us more detail. ^TS",
        "Thanks for contacting us. Here's the privacy help page — reply with specifics if needed. ^TS",
    ],
    "feature_how_to": [
        "Happy to help! You'll find steps here: https://help.twitter.com — tell us which feature if you're stuck. ^TS",
        "Hi! Try Settings → Privacy for mute/block options. Let us know if you hit a snag. ^TS",
    ],
    "spam_or_bots": [
        "Sorry about the spam. Please report the accounts and DM us a few links so we can investigate. ^TS",
        "Thanks for flagging. Report as spam, then share examples via DM. ^TS",
    ],
    "thanks_or_other": [
        "You're welcome — glad we could help! ^TS",
        "Hi! How can we help today? ^TS",
    ],
}


def generate(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    intents = list(TEMPLATES.keys())
    rows = []
    for i in range(n):
        intent = intents[i % len(intents)]
        cust = rng.choice(TEMPLATES[intent])
        # slight paraphrase noise
        if rng.random() < 0.3:
            cust = cust + " " + rng.choice(["pls", "asap", "still broken", "please help", ""])
        reply = rng.choice(REPLIES[intent])
        rows.append(
            {
                "pair_id": f"synth_{i}_{intent}",
                "customer_tweet_id": f"c{i}",
                "brand_tweet_id": f"b{i}",
                "customer_text": cust.strip(),
                "brand_reply": reply,
                "created_at": "Wed Oct 11 12:00:00 +0000 2017",
                "source": "synthetic_fallback",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    args = parser.parse_args()
    ensure_dirs()
    pairs = generate(n=args.n)
    pairs = attach_weak_labels(pairs)
    save_processed(pairs)
    print("Synthetic TwitterSupport corpus written (fallback). Prefer real Kaggle data when available.")


if __name__ == "__main__":
    main()