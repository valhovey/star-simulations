# Star Shape Simulations

I noticed some spiral shapes in the star spikes of my Newtonian telescope and wanted to understand where they were coming from. This repo uses the Poppy diffraction simulation library to simulate star shapes using various newtonian support patterns. I'm imaging on a 200mm diameter f/4 newtonian with offset vanes using an APS-C sized camera with 3.76um pixels.

## Example simulation with offset vanes and a slight misalignment on the right-hand vane:

<img width="3542" height="1827" alt="image" src="https://github.com/user-attachments/assets/6417ffbc-b89d-466d-a294-418c5ee2928a" />

## Example simulations showing centered vanes vs. offset vanes

**Centered**
<img width="2379" height="1187" alt="center" src="https://github.com/user-attachments/assets/aa948b5f-e184-4e16-a21d-d17ea8ffc57e" />

**Offset**
<img width="2379" height="1187" alt="offset" src="https://github.com/user-attachments/assets/67e0c363-1809-44b4-8593-497e1be83101" />

## Example animation showing how offset gradually changes PSF

![output](https://github.com/user-attachments/assets/3dfcb59e-0393-4d4a-9704-813189801e2c)

## Setup

This project is managed with `uv`, so once you have that set up you can `uv sync` and then `uv run main.py` to see the simulation results.
