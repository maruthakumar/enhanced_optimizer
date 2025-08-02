"""
Error Notifier for Critical Error Alerts

Sends notifications via email and Slack for critical errors that require
immediate operator attention.
"""

import os
import json
import smtplib
import logging
import traceback
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import urllib.request
import urllib.parse

class ErrorNotifier:
    """Handles error notifications via multiple channels"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize ErrorNotifier with configuration
        
        Args:
            config_file: Path to notification configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        # Initialize notification channels
        self.email_enabled = self.config.get('email', {}).get('enabled', False)
        self.slack_enabled = self.config.get('slack', {}).get('enabled', False)
        
        # Track sent notifications to prevent spam
        self.notification_history = []
        self.max_notifications_per_hour = self.config.get('rate_limit', 10)
    
    def notify_critical_error(self, error: Exception, context: Dict[str, Any],
                            severity: str = "CRITICAL") -> bool:
        """
        Send notification for critical errors
        
        Args:
            error: The exception that occurred
            context: Dictionary containing error context (parameters, state, etc.)
            severity: Error severity level
            
        Returns:
            bool: True if notification sent successfully
        """
        # Check rate limiting
        if not self._check_rate_limit():
            self.logger.warning("Notification rate limit exceeded")
            return False
        
        # Prepare error details
        error_details = self._prepare_error_details(error, context, severity)
        
        # Send notifications through enabled channels
        success = False
        
        if self.email_enabled:
            success |= self.send_email_notification(error_details)
        
        if self.slack_enabled:
            success |= self.send_slack_notification(error_details)
        
        # Log to file as backup
        self._log_to_file(error_details)
        
        # Record notification
        if success:
            self.notification_history.append({
                'timestamp': datetime.now(),
                'error_type': type(error).__name__,
                'severity': severity
            })
        
        return success
    
    def send_email_notification(self, error_details: Dict[str, Any]) -> bool:
        """
        Send email notification
        
        Args:
            error_details: Dictionary containing formatted error information
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            email_config = self.config.get('email', {})
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{error_details['severity']}] Heavy Optimizer Error: {error_details['error_type']}"
            msg['From'] = email_config.get('from_address', 'heavy-optimizer@example.com')
            msg['To'] = ', '.join(email_config.get('recipients', []))
            
            # Create email body
            text_body = self._format_email_text(error_details)
            html_body = self._format_email_html(error_details)
            
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            smtp_server = email_config.get('smtp_server', 'localhost')
            smtp_port = email_config.get('smtp_port', 587)
            smtp_use_tls = email_config.get('use_tls', True)
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if smtp_use_tls:
                    server.starttls()
                
                # Authentication if required
                username = email_config.get('username')
                password = email_config.get('password')
                if username and password:
                    server.login(username, password)
                
                server.send_message(msg)
            
            self.logger.info(f"Email notification sent for {error_details['error_type']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            return False
    
    def send_slack_notification(self, error_details: Dict[str, Any]) -> bool:
        """
        Send Slack notification
        
        Args:
            error_details: Dictionary containing formatted error information
            
        Returns:
            bool: True if Slack message sent successfully
        """
        try:
            slack_config = self.config.get('slack', {})
            webhook_url = slack_config.get('webhook_url')
            
            if not webhook_url:
                self.logger.warning("Slack webhook URL not configured")
                return False
            
            # Create Slack message
            slack_message = {
                "text": f"Heavy Optimizer Error Alert",
                "attachments": [{
                    "color": self._get_severity_color(error_details['severity']),
                    "title": f"{error_details['severity']}: {error_details['error_type']}",
                    "text": error_details['message'],
                    "fields": [
                        {
                            "title": "Job ID",
                            "value": error_details.get('job_id', 'N/A'),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": error_details['timestamp'],
                            "short": True
                        },
                        {
                            "title": "Location",
                            "value": f"{error_details['file']}:{error_details['line']}",
                            "short": False
                        }
                    ],
                    "footer": "Heavy Optimizer Platform",
                    "ts": int(datetime.now().timestamp())
                }]
            }
            
            # Add stack trace if configured
            if slack_config.get('include_stack_trace', False):
                slack_message['attachments'][0]['fields'].append({
                    "title": "Stack Trace",
                    "value": f"```{error_details['stack_trace'][:1000]}```",
                    "short": False
                })
            
            # Send to Slack
            data = json.dumps(slack_message).encode('utf-8')
            req = urllib.request.Request(webhook_url, data=data, 
                                       headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    self.logger.info(f"Slack notification sent for {error_details['error_type']}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _prepare_error_details(self, error: Exception, context: Dict[str, Any],
                              severity: str) -> Dict[str, Any]:
        """Prepare detailed error information for notifications"""
        # Get stack trace
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        stack_trace = ''.join(tb_lines)
        
        # Get error location
        if error.__traceback__:
            tb_frame = error.__traceback__.tb_frame
            file_path = tb_frame.f_code.co_filename
            line_no = error.__traceback__.tb_lineno
            function_name = tb_frame.f_code.co_name
        else:
            file_path = "Unknown"
            line_no = 0
            function_name = "Unknown"
        
        return {
            'severity': severity,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error_type': type(error).__name__,
            'message': str(error),
            'file': file_path,
            'line': line_no,
            'function': function_name,
            'stack_trace': stack_trace,
            'job_id': context.get('job_id', 'N/A'),
            'context': context,
            'system_info': self._get_system_info()
        }
    
    def _format_email_text(self, error_details: Dict[str, Any]) -> str:
        """Format error details as plain text email"""
        return f"""
Heavy Optimizer Platform - Error Notification

Severity: {error_details['severity']}
Time: {error_details['timestamp']}
Job ID: {error_details.get('job_id', 'N/A')}

Error Type: {error_details['error_type']}
Message: {error_details['message']}

Location: {error_details['file']}:{error_details['line']} in {error_details['function']}

Stack Trace:
{error_details['stack_trace']}

Context:
{json.dumps(error_details['context'], indent=2)}

System Information:
{json.dumps(error_details['system_info'], indent=2)}

This is an automated notification from the Heavy Optimizer Platform.
Please investigate this error immediately.
"""
    
    def _format_email_html(self, error_details: Dict[str, Any]) -> str:
        """Format error details as HTML email"""
        severity_color = self._get_severity_color(error_details['severity'])
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .header {{ background-color: {severity_color}; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .detail-row {{ margin: 10px 0; }}
        .label {{ font-weight: bold; }}
        .code {{ background-color: #f5f5f5; padding: 10px; font-family: monospace; }}
        .context {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Heavy Optimizer Platform - Error Notification</h2>
        <p>Severity: {error_details['severity']}</p>
    </div>
    
    <div class="content">
        <div class="detail-row">
            <span class="label">Time:</span> {error_details['timestamp']}
        </div>
        <div class="detail-row">
            <span class="label">Job ID:</span> {error_details.get('job_id', 'N/A')}
        </div>
        <div class="detail-row">
            <span class="label">Error Type:</span> {error_details['error_type']}
        </div>
        <div class="detail-row">
            <span class="label">Message:</span> {error_details['message']}
        </div>
        <div class="detail-row">
            <span class="label">Location:</span> {error_details['file']}:{error_details['line']} in {error_details['function']}
        </div>
        
        <h3>Stack Trace</h3>
        <div class="code">
            <pre>{error_details['stack_trace']}</pre>
        </div>
        
        <h3>Context</h3>
        <div class="context">
            <pre>{json.dumps(error_details['context'], indent=2)}</pre>
        </div>
        
        <p><em>This is an automated notification. Please investigate immediately.</em></p>
    </div>
</body>
</html>
"""
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color code for severity level"""
        colors = {
            'CRITICAL': '#d32f2f',
            'ERROR': '#f44336',
            'WARNING': '#ff9800',
            'INFO': '#2196f3'
        }
        return colors.get(severity.upper(), '#757575')
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information for debugging"""
        return {
            'platform': os.name,
            'python_version': os.sys.version,
            'working_directory': os.getcwd(),
            'process_id': os.getpid()
        }
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = datetime.now()
        hour_ago = current_time.replace(minute=0, second=0, microsecond=0)
        
        # Count recent notifications
        recent_notifications = [
            n for n in self.notification_history
            if n['timestamp'] >= hour_ago
        ]
        
        return len(recent_notifications) < self.max_notifications_per_hour
    
    def _log_to_file(self, error_details: Dict[str, Any]) -> None:
        """Log error details to file as backup"""
        try:
            log_dir = Path("/mnt/optimizer_share/logs/error_notifications")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"error_{timestamp}.json"
            
            with open(log_file, 'w') as f:
                json.dump(error_details, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to log error details to file: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default notification configuration"""
        return {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'use_tls': True,
                'from_address': 'heavy-optimizer@example.com',
                'recipients': ['admin@example.com'],
                'username': None,
                'password': None
            },
            'slack': {
                'enabled': False,
                'webhook_url': None,
                'include_stack_trace': True
            },
            'rate_limit': 10
        }