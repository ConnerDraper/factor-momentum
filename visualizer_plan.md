# Visualizer Upgrade Plan: Professional & Beautiful (Not AI-Generic)

## Goal
Transform the visualization scripts from "generic LLM output" to "professional quantitative researcher" aesthetic. The goal is beauty through clarity, not decoration. Think Edward Tufte or a high-quality academic paper, not a default Streamlit app.

## Aesthetic Changes

### 1. Color Palette
- **Current problem:** Random Tailwind hex codes (`#2563eb`) that look like a web app.
- **Solution:** Use a sophisticated, muted palette or semantic colors.
    - **Main series:** A deep, professional blue (e.g., `'#004488'` or `tab:blue` with slightly lower alpha).
    - **Highlights (Optimal Lambda):** A distinct but not jarring contrast (e.g., a dark red or amber, not bright neon).
    - **Background:** White or very light gray (`whitesmoke`) for the figure background, but **no internal grid lines** (or very faint ones).

### 2. Typography & Layout
- **Current problem:** Bold titles, massive suptitles, redundant labels.
- **Solution:**
    - **Titles:** Left-aligned, regular weight (not bold), slightly larger size.
    - **Axis Labels:** Concise. "Cumulative Return" instead of "Cumulative Log Returns (%)".
    - **Spine removal:** Remove the top and right spines (`despine`) to reduce visual clutter (data-ink ratio).
    - **Fonts:** Use a cleaner sans-serif font if available, or just standard DejaVu Sans but with better spacing.

### 3. Data Representation
- **Current problem:** `sns.lineplot` with default smoothing/bands, fat markers.
- **Solution:**
    - **Lines:** Thinner, sharper lines (`linewidth=1.5`).
    - **Markers:** Smaller or removed entirely if data is dense.
    - **Grid:** Remove the default heavy grid. Use faint horizontal lines *only* (`ax.yaxis.grid(True, alpha=0.15)`), no vertical lines unless necessary for time periods.
    - **DPI:** Increase to `300` for crisp, publication-ready export.

## Functional Changes

### 1. Console Output Logging
- **Requirement:** "Save the stuff it prints to console to a text file."
- **Implementation:**
    - Create a `Tee` logger or simply write to a file explicitly alongside print statements.
    - Save to `{split_dir}/report_lambda_{split}.txt`.
    - Format the output as a clean markdown table or aligned text block, avoiding the generic `====` borders.

## Detailed To-Do List

### `3_evaluate_lambdas.py`
- [ ] **Data Viz:**
    - Split into two clean subplots (Mean Return, Sharpe).
    - Use a professional style context (e.g., `plt.style.use('seaborn-v0_8-paper')` or manual rcParams).
    - Remove top/right spines.
    - Use a semantic color for the separate lines (e.g. Rate of Return = Blue, Risk/Sharpe = Green/Gray).
    - Mark the "Optimal" point with a subtle annotation or vertical line, but not a thick dashed red line.
- [ ] **Logging:**
    - Implement file writing for the results table.

### `4_visualize.py`
- [ ] **Data Viz:**
    - Drop `seaborn`. Use `matplotlib` directly for fine control.
    - Plot cumulative returns with a fill_between for drawdown (optional, but looks pro) or just a clean line.
    - Add a benchmark line (e.g., S&P 500 or just Zero line) for context if data available, otherwise just the 0 line.
    - Title: "Cumulative Return (Log Scale)" — simple and clear.
    - Add a text box in the chart area with the key stats (Sharpe, Vol, Return) so the chart stands alone.
- [ ] **Logging:**
    - Append the specific run's stats to a cumulative log file or a single run report.
