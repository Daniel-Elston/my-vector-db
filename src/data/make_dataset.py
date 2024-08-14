from __future__ import annotations

from datetime import datetime

import pandas as pd

from config.state_init import StateManager
from utils.execution import TaskExecutor


class MakeDataset:
    """Load dataset and perform base processing"""

    def __init__(self, state: StateManager):
        self.state = state
        self.dc = state.data_config

    def pipeline(self) -> pd.DataFrame:
        df = self.make_raw_set()

        steps = [
            self.add_document,
        ]
        for step in steps:
            df = TaskExecutor.run_child_step(step, df)
        return df

    def make_raw_set(self):
        stories = [
            {
                "id": 1,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "title": "Cosmic Voyager",
                "document": "The starship Nebula pierced the void, its quantum engines humming. Captain Zara studied the swirling darkness ahead—a supermassive black hole. 'Initiate gravitational shielding,' she commanded. As they crossed the event horizon, spacetime twisted. Rainbows of light bent around them, echoes of the universe they left behind. The ship's AI chirped warnings of immense tidal forces. Zara gripped her chair, heart pounding. They were making history, venturing where no human had gone before. What secrets lay beyond this cosmic maelstrom?",
            },
            {
                "id": 2,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "title": "Abyssal Luminescence",
                "document": "In the inkiest depths of the Mariana Trench, a fangtooth fish named Glow navigated by bioluminescent flashes. Crushing pressure and eternal darkness were his home. Glow's rows of needle-sharp teeth sensed vibrations—food approaching. With a burst of speed, he snapped up a viperfish. Suddenly, a colossal shape loomed. Glow recognized the undulating tentacles of a giant squid. He darted away, diving deeper into his abyssal sanctuary, where only the most adapted creatures could survive.",
            },
            {
                "id": 3,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "title": "AutoMend GT-3000",
                "document": "The AutoMend GT-3000 cruised down the solar highway, its graphene frame gleaming. Inside, Sarah relaxed as the car drove itself. A ping alerted her: 'Minor abrasion detected on rear panel.' Nanobots swarmed to the site, reconstructing the damaged area molecule by molecule. The car's AI announced, 'Repair complete. Efficiency increased by 0.02%.' Sarah smiled, remembering the old days of rust and body shops. As the GT-3000 merged lanes, its adaptive camouflage shifted, optimizing solar absorption. This wasn't just transportation; it was evolution on wheels.",
            },
        ]
        return pd.DataFrame(stories)

    def add_document(self, df):
        """Add a new document to the DataFrame."""
        if self.dc.input_title and self.dc.input_document is not None:
            new_id = df["id"].max() + 1 if not df.empty else 1
            new_row = {
                "id": new_id,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "title": self.dc.input_title,
                "document": self.dc.input_document,
            }
            return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            return df
