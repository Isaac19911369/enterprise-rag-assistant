"""Demo integrations: an in-memory Service Desk and HR system.

In a real deployment these call ServiceNow/Jira and the HR system's API with the
*user's* delegated token, so the assistant can never see more than the user can.
"""

from __future__ import annotations

import itertools
from datetime import date

from .tools import ToolRegistry

registry = ToolRegistry()

_TICKETS: dict[str, dict] = {
    "INC-1042": {
        "summary": "VPN disconnects every 10 minutes",
        "status": "In progress",
        "assignee": "Network team",
        "severity": "SEV3",
    },
    "INC-1043": {
        "summary": "Request: second monitor",
        "status": "Resolved",
        "assignee": "Workplace IT",
        "severity": "SEV3",
    },
}
_LEAVE = {"E1001": {"entitlement": 25, "taken": 11, "carried_over": 3}}
_counter = itertools.count(2001)


@registry.register()
def get_ticket_status(ticket_id: str) -> dict:
    """Look up the status of a Service Desk ticket.

    ticket_id: Ticket number such as INC-1042
    """
    ticket = _TICKETS.get(ticket_id.upper())
    return {"ticket_id": ticket_id.upper(), **ticket} if ticket else {"error": "ticket not found"}


@registry.register(
    side_effect=True,
    enum={
        "severity": ["SEV1", "SEV2", "SEV3"],
        "category": ["Hardware", "Software request", "Network / VPN", "Access", "Other"],
    },
)
def create_ticket(category: str, summary: str, severity: str = "SEV3") -> dict:
    """Open a new Service Desk ticket on behalf of the user.

    category: Service Desk category
    summary: One-line description of the problem
    severity: Incident severity, SEV3 unless a production system is affected
    """
    ticket_id = f"INC-{next(_counter)}"
    _TICKETS[ticket_id] = {
        "summary": summary,
        "status": "New",
        "assignee": "Service Desk",
        "severity": severity,
        "category": category,
    }
    return {"ticket_id": ticket_id, "status": "New"}


@registry.register()
def get_leave_balance(employee_id: str) -> dict:
    """Return the remaining annual leave days of an employee for the current year.

    employee_id: Employee number such as E1001
    """
    rec = _LEAVE.get(employee_id.upper())
    if rec is None:
        return {"error": "employee not found"}
    remaining = rec["entitlement"] + rec["carried_over"] - rec["taken"]
    return {
        "employee_id": employee_id.upper(),
        "year": date.today().year,
        "remaining_days": remaining,
        **rec,
    }
