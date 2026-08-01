# Mark-1 🏎️
This project brings forward my longtime passion for Racing and Motorsport combined with my skillset in AI/ML and Data Analysis. The aim of this project is to provide a unified data analysis and visualisation library that is capable of analysing and debriefing race weekends.

## The Four Pillars 📜
- Demystify Car Setup
- Understand Tyre Degradation and Pace
- Align Driving Styles with Car Performance
- Analyse Telemetry Channels

## Features 🧪
- `Abstraction`: The Package abstracts most / all of the underlying calls made to FastF1 thereby behaving as an orchestrator.
- `Visualisations`: The Package provides a separate visualisation suite (using Matplotlib and Seaborn) that integrates well with all of the Pillars to best illustrate the racing data.
- `Feature Engineering`: The Package performs many feature engineering operations to improve the understanding for the mechanics of the car. This plays a vital role in the downstream analytics.

## Current Progress 🕣
- The API is being ironed out for F1 races first as a combination of high-level analysis on the Laps Frames and low-level telemetry traces across multiple channels.
- On completion the API will be extended to support other data sources apart from F1 such as WEC.

