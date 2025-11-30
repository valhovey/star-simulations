# Star Shape Simulations

I noticed some spiral shapes in the star spikes of my Newtonian telescope and wanted to understand where they were coming from. This repo uses the Poppy diffraction simulation library to simulate star shapes using various newtonian support patterns. I'm imaging on a 200mm diameter f/4 newtonian with offset vanes using an APS-C sized camera with 3.76um pixels.

Example simulation with offset vanes and a slight misalignment on the right-hand vane:
<img width="3542" height="1827" alt="image" src="https://github.com/user-attachments/assets/6417ffbc-b89d-466d-a294-418c5ee2928a" />

## Setup

This project is managed with `uv`, so once you have that set up you can `uv sync` and then `uv run main.py` to see the simulation results.
