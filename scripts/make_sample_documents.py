"""Generate the sample enterprise PDF library used by the demo.

The files are synthetic but realistic: five policy documents with stable page
numbers, so the evaluation set can assert "answer on page 3 of the handbook"
and citations can be checked automatically.

Usage::

    python scripts/make_sample_documents.py
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pymupdf

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from rag_assistant.config import get_settings

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN = 72
LINE_HEIGHT = 15.5
MAX_CHARS = 88

DOCUMENTS: dict[str, list[list[str]]] = {
    "Employee_Handbook.pdf": [
        [
            "1. About this handbook",
            "This handbook summarises the employment policies of Northwind Technologies Pvt. Ltd.",
            "It applies to all permanent employees and interns from their first day.",
            "Where a policy conflicts with local law, local law applies.",
            "Questions can be sent to people@northwind.example.",
        ],
        [
            "2. Working hours and attendance",
            "Standard working hours are 9:30 to 18:30 IST, Monday to Friday.",
            "Employees may choose a flexible start between 8:30 and 10:30 with manager approval.",
            "Core collaboration hours are 11:00 to 16:00 IST; meetings should be scheduled inside this window.",
            "Attendance is tracked in the HR portal. Three late arrivals in a month require a written explanation.",
        ],
        [
            "3. Leave policy",
            "Employees receive 24 days of paid annual leave per calendar year.",
            "Sick leave: 12 days. Casual leave: 8 days. Leave is credited pro-rata in the first year.",
            "Annual leave accrues monthly at 2 days per month and up to 10 unused days may be carried forward.",
            "Leave must be applied for at least 5 working days in advance, except in emergencies.",
            "Unused casual leave lapses at the end of the calendar year.",
        ],
        [
            "4. Code of conduct",
            "Employees must treat colleagues, customers and partners with respect.",
            "Gifts above INR 5,000 must be disclosed to the compliance team.",
            "Conflicts of interest must be declared to your manager and to compliance@northwind.example.",
            "Confidential information must not be shared outside the company without written approval.",
        ],
        [
            "5. Grievance and POSH",
            "Raise a grievance with your manager or HR business partner. Escalations go to the Grievance Committee.",
            "The Internal Committee (IC) handles sexual harassment complaints under the POSH Act.",
            "Complaints can be sent to ic@northwind.example and are treated confidentially.",
            "The Internal Committee aims to complete an inquiry within 90 days.",
        ],
        [
            "6. Separation and notice period",
            "The notice period is 60 days for permanent employees and 15 days during probation.",
            "Notice must be submitted in writing through the HR portal.",
            "Full and final settlement is processed within 45 days of the last working day.",
            "Exit interviews are mandatory for employees with more than six months of tenure.",
        ],
    ],
    "Remote_Work_Policy.pdf": [
        [
            "1. Purpose and scope",
            "This policy explains hybrid and remote working arrangements at Northwind Technologies.",
            "It applies to all employees whose role is marked as remote-eligible in the HR system.",
            "Remote work is a privilege, not an entitlement, and can be withdrawn with 30 days notice.",
        ],
        [
            "2. Hybrid schedule and approval",
            "Remote-eligible employees may work remotely for two days per week.",
            "The remaining three days are spent in the office; Tuesday and Thursday are mandatory office days.",
            "Employees must request recurring remote days through the HR portal at least 7 days in advance.",
            "Managers may approve up to five remote days per week for exceptional cases such as medical needs.",
        ],
        [
            "3. Home office allowance",
            "New employees receive a one-time home office allowance of INR 25,000.",
            "The allowance covers a desk, chair, monitor and internet setup.",
            "Claims must be submitted with invoices within 60 days of joining.",
            "Internet reimbursement is capped at INR 1,500 per month for approved remote employees.",
        ],
        [
            "4. Security and cross-border work",
            "Remote work must use the company VPN and an encrypted laptop.",
            "Working from outside India requires prior approval from HR and the security team.",
            "Cross-border approval is valid for a maximum of 30 days per calendar year.",
            "Employees must report a lost or stolen device within one hour to security@northwind.example.",
        ],
    ],
    "IT_Security_Policy.pdf": [
        [
            "1. Scope",
            "This policy applies to every employee, contractor and vendor using Northwind systems.",
            "Non-compliance can lead to disciplinary action and loss of system access.",
        ],
        [
            "2. Passwords and authentication",
            "Passwords must be at least 14 characters long and must not be reused across services.",
            "Multi-factor authentication (MFA) is mandatory for email, VPN and the HR portal.",
            "Passwords must be rotated every 90 days for privileged accounts.",
            "Never share passwords or MFA codes with anyone, including IT support.",
        ],
        [
            "3. Devices and network",
            "Only company-managed and encrypted laptops may access production systems.",
            "Personal devices may access email only through the mobile device management profile.",
            "USB storage devices are blocked by default; exceptions need security approval.",
            "Public Wi-Fi requires the company VPN and split tunnelling is disabled.",
        ],
        [
            "4. Incident reporting",
            "Report suspected phishing, malware or data loss within one hour to security@northwind.example.",
            "Do not investigate an incident yourself; preserve evidence and disconnect the device.",
            "The security team acknowledges severity 1 incidents within 15 minutes, 24 hours a day.",
            "A post-incident review is completed within 5 business days.",
        ],
        [
            "5. Data classification and access",
            "Data is classified as Public, Internal, Confidential and Restricted.",
            "Production customer data is Restricted and requires manager plus security approval.",
            "Access is reviewed every quarter; unused access is revoked after 30 days.",
            "Sharing Restricted data outside the company requires a signed data processing agreement.",
        ],
    ],
    "Expense_and_Travel_Policy.pdf": [
        [
            "1. Scope and principles",
            "These rules cover business travel, meals and reimbursable expenses.",
            "Spend company money as if it were your own: necessary, reasonable and documented.",
            "Personal expenses, alcohol and traffic fines are not reimbursable.",
        ],
        [
            "2. Travel limits",
            "Domestic hotel limit: INR 6,000 per night in metro cities and INR 4,000 elsewhere.",
            "International hotel limit: USD 180 per night unless the city is on the approved exception list.",
            "Economy class is standard for flights under six hours; premium economy needs VP approval.",
            "Meal allowance is INR 1,500 per day for domestic travel.",
        ],
        [
            "3. Approval and reimbursement",
            "Travel must be approved in the HR portal before booking.",
            "Receipts must be submitted within 15 days of returning from travel.",
            "Claims older than 60 days require finance director approval.",
            "Reimbursements are paid with the next payroll cycle after approval.",
        ],
        [
            "4. Corporate card",
            "The corporate card is for business expenses only and must never be shared.",
            "Lost cards must be reported to finance@northwind.example immediately.",
            "Cash withdrawals on the corporate card are disabled.",
            "Monthly card statements must be reconciled by the 5th of the following month.",
        ],
    ],
    "Product_Support_SLA.pdf": [
        [
            "1. Support tiers",
            "RAG Knowledge Assistant offers Standard, Business and Enterprise support plans.",
            "Enterprise customers receive a dedicated technical account manager.",
            "Support is available 24 hours a day for severity 1 issues on Business and Enterprise plans.",
        ],
        [
            "2. Severity definitions",
            "Severity 1 (P1): production unavailable or data loss affecting all users.",
            "Severity 2 (P2): a major feature is unavailable with no workaround.",
            "Severity 3 (P3): degraded performance or a workaround exists.",
            "Severity 4 (P4): cosmetic issue or feature request.",
        ],
        [
            "3. Response and resolution targets",
            "P1 first response within 15 minutes and a workaround target of 4 hours.",
            "P2 first response within 1 hour and resolution target of 1 business day.",
            "P3 first response within 8 business hours and resolution target of 5 business days.",
            "P4 first response within 2 business days; resolution is scheduled in a future release.",
        ],
        [
            "4. Escalation",
            "If a P1 is not acknowledged in 15 minutes, escalate to the on-call support manager.",
            "Unresolved P2 issues escalate to the engineering manager after 4 hours.",
            "Commercial disputes are handled by the customer success director.",
            "Post-incident reports are shared within 5 business days for all P1 incidents.",
        ],
    ],
}


def wrap(text: str, width: int = MAX_CHARS) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [""]


def write_pdf(path: Path, title: str, pages: list[list[str]]) -> int:
    document = pymupdf.open()
    for page_number, paragraphs in enumerate(pages, start=1):
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y = MARGIN
        page.insert_text((MARGIN, y), title, fontsize=17, fontname="hebo")
        y += 12
        page.draw_line((MARGIN, y), (PAGE_WIDTH - MARGIN, y), width=0.8)
        y += 26

        for paragraph in paragraphs:
            for line in wrap(paragraph):
                if y > PAGE_HEIGHT - MARGIN:
                    raise ValueError(
                        f"{title} page {page_number} overflows; shorten the sample text."
                    )
                page.insert_text((MARGIN, y), line, fontsize=10.5, fontname="helv")
                y += LINE_HEIGHT
            y += 6

        page.insert_text(
            (PAGE_WIDTH - MARGIN - 40, PAGE_HEIGHT - 30),
            f"Page {page_number} of {len(pages)}",
            fontsize=8,
            color=(0.4, 0.4, 0.4),
        )

    document.set_metadata({"title": title, "author": "Northwind Technologies"})
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    document.close()
    return len(pages)


def create_sample_documents(output_dir: Path | None = None) -> list[Path]:
    target = output_dir or get_settings().sample_docs_dir
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename, pages in DOCUMENTS.items():
        path = target / filename
        page_count = write_pdf(path, Path(filename).stem.replace("_", " "), pages)
        written.append(path)
        print(f"  {filename:<34} {page_count} pages  {path.stat().st_size / 1024:.1f} KB")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the sample policy PDF library.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    target = Path(args.output_dir) if args.output_dir else get_settings().sample_docs_dir
    print(f"Writing sample documents to {target}")
    create_sample_documents(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
