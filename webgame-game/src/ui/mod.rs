pub mod menu_button;

use bevy::prelude::*;
use menu_button::MenuButtonPlugin;

/// Manages UI logic and components.
pub struct UIPlugin;

impl Plugin for UIPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MenuButtonPlugin);
    }
}