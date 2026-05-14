#!/bin/bash
# Monitor and auto-fix ReviewForge pipeline
# Run every 20 minutes via cron

LOG="/tmp/rf_monitor.log"
cd /mnt/e/Documents/github_clone/ReviewForge

echo "=== $(date) ===" >> $LOG

# 1. Syntax check
echo "[1/4] Syntax check..." >> $LOG
.venv/bin/python -m py_compile src/retrievers/serpapi.py src/retrievers/base.py src/retrievers/arxiv.py src/llm.py 2>&1 >> $LOG
if [ $? -eq 0 ]; then
    echo "  ✓ Syntax OK" >> $LOG
else
    echo "  ✗ Syntax errors, fixing..." >> $LOG
    /root/.openclaw/npm/node_modules/.bin/acpx claude exec "修复语法错误" >> $LOG 2>&1
fi

# 2. Network test
echo "[2/4] Network connectivity test..." >> $LOG
.venv/bin/python scripts/test_tool_env.py >> $LOG 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ test_tool_env.py passed" >> $LOG
else
    echo "  ✗ test_tool_env.py failed" >> $LOG
fi

# 3. Streamlit process check
echo "[3/4] Streamlit process check..." >> $LOG
if pgrep -f "streamlit run" > /dev/null; then
    echo "  ✓ Streamlit running" >> $LOG
else
    echo "  ✗ Streamlit not running, restarting..." >> $LOG
    nohup .venv/bin/python -m streamlit run ui/app.py --server.port 8501 >> $LOG 2>&1 &
fi

# 4. Pipeline smoke test
echo "[4/4] Pipeline smoke test..." >> $LOG
.venv/bin/python scripts/run_pipeline.py "测试" >> $LOG 2>&1 &
PIPELINE_PID=$!
sleep 30
if kill -0 $PIPELINE_PID 2>/dev/null; then
    echo "  ✓ Pipeline started (PID $PIPELINE_PID)" >> $LOG
else
    echo "  ✗ Pipeline exited early" >> $LOG
fi

echo "Done at $(date)" >> $LOG
