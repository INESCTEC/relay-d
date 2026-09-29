<div align="center">

![logo](./img/inesctec_banner.png)

## INESC TEC
### Open-Source Software: *Security*

</div>

### Reporting security issues

> [!IMPORTANT]
> **Please do not publish security vulnerabilities on public GitHub issues.**

Any vulnerability found should be reported to the Organization Stewards (contacts available in the repository README file).
If you are still waiting for a response within 48 hours, please send us a follow-up email to confirm that we received your initial message.

### Reporting languages

> [!NOTE]
> Please write all your reports in English.

### Security Best Practices

> [!NOTE]
**We encourage all contributors to follow security best practices when contributing code.**

- **Validate Input**: Always sanitize and validate user input to prevent injection attacks (SQL injection, XSS, etc.).
- **Use Secure Protocols**: Prefer using HTTPS over HTTP for all communication.
- **Minimize Permissions**: Follow the principle of least privilege when assigning permissions to users and components.
- **Regularly Update Dependencies**: Ensure that all dependencies are up-to-date and free from known vulnerabilities.

### Security Vulnerability Lifecycle

> [!NOTE]
**Here is an overview of the typical process when handling security vulnerabilities:**

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
- **Assessment**: Our team will assess the severity and potential impact of the vulnerability.
- **Fix and Mitigation**: We will implement a fix or mitigation strategy and document it in the changelog.
- **Disclosure**: After the fix is deployed, we will provide a public disclosure with details of the vulnerability, its severity, and the fix.

### Security Fixes and Changelog Updates

> [!IMPORTANT]
> **Once a security issue is resolved or mitigated, it is essential to indicate the fix in the changelog following best practices.**

When a security vulnerability is addressed, it should be recorded in the project's changelog with the following details:

- **Severity Level**: Indicate the severity of the issue (e.g., moderate, high, critical).
- **Publication Timeline**: Provide an estimated release date for the security fix, typically 30 to 60 days after the code is made public.

This ensures transparency and informs users about the actions taken to mitigate security risks.

---

### Reporting Template

For reporting security vulnerabilities, please use the [Security Issue Reporting Template](../docs/reporting_template.md) to provide all relevant details. This helps us address the issue more efficiently and ensures that we capture all the necessary information to resolve it.
