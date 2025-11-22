
import json
import numpy as np
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import os

def generate_voronoi(json_path, output_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Use the first available frame
        first_frame_key = sorted(list(data.keys()))[0]
        coords = data[first_frame_key]["Coordinates"]

        pitch = Pitch(pitch_type="uefa", pitch_color="None", goal_type="box", linewidth=0.8)
        fig, ax = pitch.draw()
        fig.set_facecolor("black")

        player_locs = []
        teams = []
        opp_locs = []
        all_x = []
        all_y = []

        for item in coords:
            id_val = item.get("ID")
            # Fix: Coordinates come as a list [x, y]
            x, y = item["Coordinates"]
            player_type = item.get("Type", None)

            if id_val == "Ball":
                ax.scatter(x, y, color="white", zorder=15, facecolors="none", edgecolors="white", s=50)
            else:
                if player_type == "Goalkeeper":
                    pass # Skip GK for Voronoi regions usually, or handle separately
                else:
                    team = item.get("Team")
                    if team is not None:
                        if team == 0:
                            color = "#add8e6" # Blue-ish
                            opp_locs.append((x, y, color))
                            teams.append(1)
                        else:
                            color = "red"
                            player_locs.append((x, y, color))
                            teams.append(0)

                        all_x.append(x)
                        all_y.append(y)

        if len(all_x) > 2:
            t1, t2 = pitch.voronoi(all_x, all_y, teams=teams)
            t1 = pitch.polygon(t1, facecolor="#add8e6", edgecolor="#add8e6", alpha=0.2, zorder=1, ax=ax)
            t2 = pitch.polygon(t2, facecolor="red", edgecolor="red", alpha=0.2, zorder=1, ax=ax)

        if player_locs:
            pitch.scatter([x[0] for x in player_locs], [x[1] for x in player_locs], color=[x[2] for x in player_locs], zorder=5, s=100, edgecolors=[x[2] for x in player_locs], ax=ax)
        if opp_locs:
            pitch.scatter([x[0] for x in opp_locs], [x[1] for x in opp_locs], color=[x[2] for x in opp_locs], zorder=5, s=100, edgecolors=[x[2] for x in opp_locs], ax=ax)

        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"Error generating Voronoi: {e}")
        return False

def generate_pass_plot(json_path, output_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Find start and end of ball
        # Just take first and last frame for simplicity
        keys = sorted(list(data.keys()))
        if not keys: return False

        start_frame = data[keys[0]]
        end_frame = data[keys[-1]]

        start_ball = None
        end_ball = None

        pitch = Pitch(pitch_type="uefa", pitch_color="None", goal_type="box", linewidth=0.8)
        fig, ax = pitch.draw()
        fig.set_facecolor("black")

        # Draw players from start frame
        for item in start_frame["Coordinates"]:
            x, y = item["Coordinates"]
            id_val = item.get("ID")
            if id_val == "Ball":
                start_ball = (x, y)
                ax.scatter(x, y, color="white", zorder=5, facecolors="none", edgecolors="white", s=50)
            else:
                team = item.get("Team")
                color = "white"
                if team == 0: color = "red"
                elif team == 1: color = "blue"
                ax.scatter(x, y, color=color, zorder=5, s=100, alpha=0.5, edgecolors=color)

        # Find end ball position
        for item in end_frame["Coordinates"]:
            if item.get("ID") == "Ball":
                end_ball = tuple(item["Coordinates"])
                break

        if start_ball and end_ball:
             ax.arrow(start_ball[0], start_ball[1], end_ball[0] - start_ball[0], end_ball[1] - start_ball[1], head_width=1, head_length=1, fc="white", ec="white", zorder=5)

        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"Error generating Pass Plot: {e}")
        return False

def generate_trajectory(json_path, output_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        pitch = Pitch(pitch_type="uefa", pitch_color="None", goal_type="box", linewidth=0.8)
        fig, ax = pitch.draw()
        fig.set_facecolor("black")

        ball_coords = []
        keys = sorted(list(data.keys()))
        for k in keys:
            frame_data = data[k]
            for item in frame_data["Coordinates"]:
                if item.get("ID") == "Ball":
                    ball_coords.append(item["Coordinates"])

        if len(ball_coords) > 1:
            ax.plot([x[0] for x in ball_coords], [x[1] for x in ball_coords], color="white", zorder=5, linewidth=1)
            ax.scatter(ball_coords[0][0], ball_coords[0][1], color="blue", zorder=5, s=50)
            ax.scatter(ball_coords[-1][0], ball_coords[-1][1], color="blue", zorder=5, s=50)

        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"Error generating Trajectory: {e}")
        return False
