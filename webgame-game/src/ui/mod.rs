pub mod menu_button;
pub mod screen_transition;
pub mod input_prompt;

use bevy::prelude::*;
use input_prompt::InputPromptPlugin;
use menu_button::MenuButtonPlugin;
use screen_transition::ScreenTransitionPlugin;

use crate::screens::ScreenState;

/// Manages UI logic and components.
pub struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((MenuButtonPlugin, ScreenTransitionPlugin::<ScreenState>::default(), InputPromptPlugin));
    }
}
