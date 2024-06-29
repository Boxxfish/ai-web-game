#![allow(clippy::type_complexity)]
#![feature(iter_array_chunks)]

use bevy::prelude::*;
#[cfg(feature = "editor")]
use bevy_editor_pls::EditorPlugin;
use configs::ReleaseCfgPlugin;

mod net;
mod configs;
mod gridworld;
mod observer;
mod world_objs;

/// Main entry point for our game.
fn main() {
    let mut app = App::new();
    app.add_plugins(ReleaseCfgPlugin);
    #[cfg(feature = "editor")]
    app.add_plugins(EditorPlugin::default());
    app.run();
}
