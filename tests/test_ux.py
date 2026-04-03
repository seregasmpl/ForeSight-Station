# tests/test_ux.py
import pytest
import httpx
from playwright.sync_api import Page, expect

BASE = "http://localhost:8001"


@pytest.fixture(autouse=True)
def reset_sessions():
    """Reset all sessions before each UX test."""
    httpx.post(f"{BASE}/api/sessions/reset-all", timeout=10)


# U1: normal flow — join, ask question, get answer (requires live LLM)
@pytest.mark.skip(reason="requires live LLM — run manually")
def test_U1_normal_flow(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    expect(page.locator("#mission-topic")).to_contain_text("Человечество")
    page.fill("#question-input", "Будут ли люди жить на Марсе?")
    page.click("#send-btn")
    expect(page.locator(".bubble-ai")).to_be_visible(timeout=90000)
    expect(page.locator(".bubble-ai").first).not_to_be_empty()


# U2: joining already-active session shows error (409)
def test_U2_double_join_shows_error(page: Page):
    httpx.post(f"{BASE}/api/sessions/join", json={"code": "ЛУНА-01"}, timeout=5)
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    expect(page.locator("#error-msg")).to_be_visible(timeout=3000)
    expect(page.locator("#error-msg")).to_contain_text("УЖЕ НА БОРТУ")


# U3: navigating to /session directly without sessionStorage redirects to /
def test_U3_direct_session_url_redirects(page: Page):
    page.goto(f"{BASE}/session")
    page.wait_for_url(f"{BASE}/", timeout=3000)
    expect(page.locator("#session-code")).to_be_visible()


# U4: wrong session code shows error
def test_U4_wrong_code_shows_error(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "МАРС-99")
    page.click("button.retro-btn")
    expect(page.locator("#error-msg")).to_be_visible(timeout=3000)
    expect(page.locator("#error-msg")).to_contain_text("НЕ НАЙДЕН")


# U5: clicking send without text does nothing
def test_U5_send_without_text(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    page.click("#send-btn")
    expect(page.locator(".bubble-thinking")).not_to_be_visible(timeout=1000)


# U6: send button becomes disabled after first click
def test_U6_double_send_blocked(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    page.fill("#question-input", "Первый вопрос?")
    page.click("#send-btn")
    expect(page.locator("#send-btn")).to_be_disabled(timeout=500)


# U7: browser back button after join returns to home page
def test_U7_back_button_after_join(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    page.go_back()
    expect(page.locator("#session-code")).to_be_visible(timeout=3000)


# U8: /admin page requires PIN
def test_U8_admin_requires_pin(page: Page):
    page.goto(f"{BASE}/admin")
    expect(page.locator("#pin-overlay")).to_be_visible()
    expect(page.locator("#dashboard")).not_to_be_visible()


# U9: emoji and special characters in question
def test_U9_emoji_in_question(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    page.fill("#question-input", "🚀 Полетим на Марс? 👾 ??? !!!")
    page.click("#send-btn")
    expect(page.locator(".bubble-user")).to_be_visible(timeout=2000)
    expect(page.locator(".bubble-user")).to_contain_text("🚀")


# U10: spaces-only input is not sent
def test_U10_spaces_only(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    page.fill("#question-input", "   ")
    page.click("#send-btn")
    expect(page.locator(".bubble-thinking")).not_to_be_visible(timeout=1000)


# U11: /viewer works without PIN and shows 8 session cards
def test_U11_viewer_no_pin(page: Page):
    page.goto(f"{BASE}/viewer")
    expect(page.locator("#sessions-grid")).to_be_visible()
    page.wait_for_selector(".session-card", timeout=5000)
    cards = page.locator(".session-card").count()
    assert cards == 8, f"Expected 8 session cards, got {cards}"


# U12: second user with same code sees 409 error
def test_U12_second_user_same_code(page: Page, browser):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    page2 = browser.new_page()
    page2.goto(BASE)
    page2.fill("#session-code", "ЛУНА-01")
    page2.click("button.retro-btn")
    expect(page2.locator("#error-msg")).to_be_visible(timeout=3000)
    expect(page2.locator("#error-msg")).to_contain_text("УЖЕ НА БОРТУ")
    page2.close()


# U13: very long question in input does not break layout
def test_U13_long_question_in_input(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    long_q = "Что будет " * 100
    page.fill("#question-input", long_q)
    expect(page.locator("#question-input")).to_be_visible()
    expect(page.locator("#send-btn")).to_be_visible()


# U14: scroll works after long LLM response (requires live LLM)
@pytest.mark.skip(reason="requires live LLM — run manually")
def test_U14_scroll_after_long_response(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    page.fill("#question-input", "Расскажи подробно о будущем освоения космоса к 2100 году")
    page.click("#send-btn")
    expect(page.locator(".bubble-ai")).to_be_visible(timeout=90000)
    last_bubble = page.locator(".bubble-ai").last
    expect(last_bubble).to_be_in_viewport(timeout=2000)


# U15: chat-area has scrollable overflow
def test_U15_chat_area_scrollable(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    overflow = page.evaluate(
        "() => getComputedStyle(document.getElementById('chat-area')).overflowY"
    )
    assert overflow in ("auto", "scroll"), f"chat-area overflow-y = {overflow!r}, expected auto/scroll"


# U16: reloading page after join redirects to home (no sessionStorage after reload)
def test_U16_reload_loses_session(page: Page):
    page.goto(BASE)
    page.fill("#session-code", "ЛУНА-01")
    page.click("button.retro-btn")
    page.wait_for_url(f"{BASE}/session", timeout=5000)
    # Simulate closing and reopening (new page = no sessionStorage)
    page2 = page.context.new_page()
    page2.goto(f"{BASE}/session")
    page2.wait_for_url(f"{BASE}/", timeout=3000)
    expect(page2.locator("#session-code")).to_be_visible()
    page2.close()
