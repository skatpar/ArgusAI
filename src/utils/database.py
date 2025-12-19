"""
SQLite Database Utility for Persistent Storage
Store model metadata, rules, logs, query history, and session data
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd


class ArgusDatabase:
    """Persistent SQLite database for ArgusAI platform"""

    def __init__(self, db_path: str = None):
        """Initialize database connection"""
        if db_path is None:
            # Default to ArgusAI root directory
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'argus_data.db'
            )

        self.db_path = db_path
        self.conn = None
        self.init_database()

    def init_database(self):
        """Initialize database and create tables if they don't exist"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries

        cursor = self.conn.cursor()

        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                model_path TEXT NOT NULL,
                model_type TEXT,
                algorithm TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metrics TEXT,  -- JSON
                hyperparameters TEXT,  -- JSON
                features TEXT,  -- JSON
                training_info TEXT,  -- JSON
                is_active BOOLEAN DEFAULT 0,
                notes TEXT
            )
        """)

        # Rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT UNIQUE NOT NULL,
                rule_type TEXT,
                condition TEXT,  -- JSON
                action TEXT,  -- JSON
                priority INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                description TEXT,
                metadata TEXT  -- JSON
            )
        """)

        # Query history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_type TEXT,  -- 'current' or 'baseline'
                row_count INTEGER,
                column_count INTEGER,
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_time_ms INTEGER,
                data_cached BOOLEAN DEFAULT 0,
                cached_data BLOB,  -- Pickled DataFrame
                status TEXT,  -- 'success' or 'error'
                error_message TEXT
            )
        """)

        # Activity logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                module TEXT,
                action TEXT,
                details TEXT,  -- JSON
                user TEXT,
                status TEXT
            )
        """)

        # Session state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                last_page TEXT,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                state_data TEXT,  -- JSON
                computation_engine TEXT DEFAULT 'pandas'
            )
        """)

        # Training runs table (MLflow-like)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                model_name TEXT,
                algorithm TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds INTEGER,
                parameters TEXT,  -- JSON
                metrics TEXT,  -- JSON
                artifacts TEXT,  -- JSON (paths to artifacts)
                status TEXT,  -- 'running', 'completed', 'failed'
                error_message TEXT
            )
        """)

        # Model deployments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                deployment_name TEXT,
                environment TEXT,  -- 'dev', 'staging', 'production'
                endpoint_url TEXT,
                deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deployed_by TEXT,
                status TEXT,  -- 'active', 'inactive', 'deprecated'
                version TEXT,
                metadata TEXT,  -- JSON
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
        """)

        self.conn.commit()

    # ==================== MODEL METHODS ====================

    def add_model(self, model_name: str, model_path: str, **kwargs) -> int:
        """Add a new model to the database"""
        cursor = self.conn.cursor()

        # Convert dict/list fields to JSON
        metrics = json.dumps(kwargs.get('metrics', {}))
        hyperparameters = json.dumps(kwargs.get('hyperparameters', {}))
        features = json.dumps(kwargs.get('features', []))
        training_info = json.dumps(kwargs.get('training_info', {}))

        cursor.execute("""
            INSERT INTO models (
                model_name, model_path, model_type, algorithm,
                metrics, hyperparameters, features, training_info, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                model_path=excluded.model_path,
                updated_at=CURRENT_TIMESTAMP,
                metrics=excluded.metrics,
                hyperparameters=excluded.hyperparameters,
                features=excluded.features,
                training_info=excluded.training_info
        """, (
            model_name, model_path,
            kwargs.get('model_type'),
            kwargs.get('algorithm'),
            metrics, hyperparameters, features, training_info,
            kwargs.get('notes')
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_models(self, active_only: bool = False) -> List[Dict]:
        """Get all models from database"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM models"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY updated_at DESC"

        cursor.execute(query)
        rows = cursor.fetchall()

        models = []
        for row in rows:
            model = dict(row)
            # Parse JSON fields
            for field in ['metrics', 'hyperparameters', 'features', 'training_info']:
                if model.get(field):
                    try:
                        model[field] = json.loads(model[field])
                    except:
                        model[field] = {}
            models.append(model)

        return models

    def set_active_model(self, model_name: str):
        """Set a model as active (deactivate others)"""
        cursor = self.conn.cursor()

        # Deactivate all
        cursor.execute("UPDATE models SET is_active = 0")

        # Activate selected
        cursor.execute("UPDATE models SET is_active = 1 WHERE model_name = ?", (model_name,))

        self.conn.commit()

    # ==================== RULES METHODS ====================

    def add_rule(self, rule_name: str, rule_type: str, condition: Dict, action: Dict, **kwargs) -> int:
        """Add a new rule"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO rules (
                rule_name, rule_type, condition, action, priority,
                is_active, created_by, description, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rule_name) DO UPDATE SET
                condition=excluded.condition,
                action=excluded.action,
                priority=excluded.priority,
                is_active=excluded.is_active,
                updated_at=CURRENT_TIMESTAMP
        """, (
            rule_name, rule_type,
            json.dumps(condition),
            json.dumps(action),
            kwargs.get('priority', 0),
            kwargs.get('is_active', True),
            kwargs.get('created_by'),
            kwargs.get('description'),
            json.dumps(kwargs.get('metadata', {}))
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_rules(self, active_only: bool = True) -> List[Dict]:
        """Get all rules"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM rules"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY priority DESC, created_at DESC"

        cursor.execute(query)
        rows = cursor.fetchall()

        rules = []
        for row in rows:
            rule = dict(row)
            # Parse JSON fields
            for field in ['condition', 'action', 'metadata']:
                if rule.get(field):
                    try:
                        rule[field] = json.loads(rule[field])
                    except:
                        rule[field] = {}
            rules.append(rule)

        return rules

    def delete_rule(self, rule_name: str):
        """Delete a rule"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM rules WHERE rule_name = ?", (rule_name,))
        self.conn.commit()

    # ==================== QUERY HISTORY METHODS ====================

    def add_query_history(self, query_text: str, query_type: str, row_count: int,
                         column_count: int, **kwargs) -> int:
        """Add query to history"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO query_history (
                query_text, query_type, row_count, column_count,
                execution_time_ms, data_cached, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_text, query_type, row_count, column_count,
            kwargs.get('execution_time_ms', 0),
            kwargs.get('data_cached', False),
            kwargs.get('status', 'success'),
            kwargs.get('error_message')
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM query_history
            ORDER BY executed_at DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    # ==================== ACTIVITY LOG METHODS ====================

    def log_activity(self, module: str, action: str, details: Dict = None,
                     user: str = None, status: str = 'success'):
        """Log an activity"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO activity_logs (module, action, details, user, status)
            VALUES (?, ?, ?, ?, ?)
        """, (module, action, json.dumps(details or {}), user, status))

        self.conn.commit()

    def get_activity_logs(self, limit: int = 100, module: str = None) -> List[Dict]:
        """Get activity logs"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM activity_logs"
        params = []

        if module:
            query += " WHERE module = ?"
            params.append(module)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        logs = []
        for row in cursor.fetchall():
            log = dict(row)
            if log.get('details'):
                try:
                    log['details'] = json.loads(log['details'])
                except:
                    log['details'] = {}
            logs.append(log)

        return logs

    # ==================== SESSION STATE METHODS ====================

    def save_session_state(self, session_id: str, last_page: str,
                          state_data: Dict, computation_engine: str = 'pandas'):
        """Save session state"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO session_state (session_id, last_page, state_data, computation_engine)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_page=excluded.last_page,
                state_data=excluded.state_data,
                computation_engine=excluded.computation_engine,
                last_active=CURRENT_TIMESTAMP
        """, (session_id, last_page, json.dumps(state_data), computation_engine))

        self.conn.commit()

    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Get session state"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM session_state WHERE session_id = ?
        """, (session_id,))

        row = cursor.fetchone()
        if row:
            state = dict(row)
            if state.get('state_data'):
                try:
                    state['state_data'] = json.loads(state['state_data'])
                except:
                    state['state_data'] = {}
            return state
        return None

    # ==================== TRAINING RUNS METHODS ====================

    def add_training_run(self, run_name: str, model_name: str, algorithm: str,
                        parameters: Dict, **kwargs) -> int:
        """Add a training run"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO training_runs (
                run_name, model_name, algorithm, started_at,
                parameters, metrics, artifacts, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_name, model_name, algorithm,
            kwargs.get('started_at', datetime.now()),
            json.dumps(parameters),
            json.dumps(kwargs.get('metrics', {})),
            json.dumps(kwargs.get('artifacts', {})),
            kwargs.get('status', 'running')
        ))

        self.conn.commit()
        return cursor.lastrowid

    def update_training_run(self, run_id: int, **kwargs):
        """Update a training run"""
        cursor = self.conn.cursor()

        updates = []
        params = []

        if 'metrics' in kwargs:
            updates.append("metrics = ?")
            params.append(json.dumps(kwargs['metrics']))

        if 'status' in kwargs:
            updates.append("status = ?")
            params.append(kwargs['status'])

        if 'completed_at' in kwargs:
            updates.append("completed_at = ?")
            params.append(kwargs['completed_at'])

        if 'duration_seconds' in kwargs:
            updates.append("duration_seconds = ?")
            params.append(kwargs['duration_seconds'])

        if updates:
            params.append(run_id)
            cursor.execute(f"UPDATE training_runs SET {', '.join(updates)} WHERE id = ?", params)
            self.conn.commit()

    def get_training_runs(self, limit: int = 50) -> List[Dict]:
        """Get training runs"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM training_runs
            ORDER BY started_at DESC
            LIMIT ?
        """, (limit,))

        runs = []
        for row in cursor.fetchall():
            run = dict(row)
            for field in ['parameters', 'metrics', 'artifacts']:
                if run.get(field):
                    try:
                        run[field] = json.loads(run[field])
                    except:
                        run[field] = {}
            runs.append(run)

        return runs

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup"""
        self.close()


# Global database instance
_db_instance = None

def get_database() -> ArgusDatabase:
    """Get or create database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = ArgusDatabase()
    return _db_instance
