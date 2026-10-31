---
title: Remote Access and VPN
owner: IT Operations
groups: [all]
---
# Remote Access and VPN

## When the VPN is required
Internal applications (the ERP, the build farm and the file shares) are only reachable through the corporate VPN. Microsoft 365, the HR portal and the Service Desk portal are internet-facing and do not require the VPN.

## Client and connection
Northwind uses the GlobalProtect client, which is pre-installed on all managed laptops. Choose the gateway closest to you: "vie-gw" for Vienna, "lnz-gw" for Linz and "muc-gw" for Munich. Sessions are disconnected automatically after 12 hours and must be re-authenticated with your hardware key.

## Working from abroad
Working remotely from outside the EU for more than 10 working days per calendar year requires prior approval from your manager and from Information Security, because of export-control and data-protection rules. Short business trips do not need this approval.

## Troubleshooting
If the VPN fails to connect, first check that the system clock is correct and restart the GlobalProtect service. If the problem persists, open a ticket in the Service Desk portal with the category "Network / VPN".
