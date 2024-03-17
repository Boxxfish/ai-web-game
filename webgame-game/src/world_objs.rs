use crate::{gridworld::GRID_CELL_SIZE, observer::Wall};
use bevy::prelude::*;

/// Plugin for world objects (e.g. doors, noise sources).
pub struct WorldObjPlugin;

impl Plugin for WorldObjPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_door);
    }
}

/// Adds playable functionality for `WorldObjPlugin`.
pub struct WorldObjPlayPlugin;

impl Plugin for WorldObjPlayPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, visualize_door);
    }
}

/// A door that can be opened and closed.
#[derive(Component, Default)]
pub struct Door {
    pub open: bool,
}

/// Opens and closes the door.
fn update_door(mut commands: Commands, door_query: Query<(Entity, &Door), Changed<Door>>) {
    for (e, door) in door_query.iter() {
        if door.open {
            commands.entity(e).remove::<Wall>();
        } else {
            commands.entity(e).insert(Wall);
        }
    }
}

/// Updates the door visual.
fn visualize_door(
    mut commands: Commands,
    mut door_query: Query<(Entity, &Door, Option<&mut Sprite>), Changed<Door>>,
) {
    for (e, door, sprite) in door_query.iter_mut() {
        if door.open {
            sprite.unwrap().color.set_a(0.5);
            commands.entity(e).remove::<Wall>();
        } else if let Some(mut sprite) = sprite {
            sprite.color.set_a(1.);
            commands.entity(e).insert(Wall);
        } else {
            commands.entity(e).insert((
                Sprite {
                    color: Color::MAROON,
                    custom_size: Some(Vec2::ONE * GRID_CELL_SIZE),
                    ..default()
                },
                Handle::<Image>::default(),
                Visibility::Visible,
                InheritedVisibility::default(),
                ViewVisibility::default(),
            ));
            commands.entity(e).insert(Wall);
        }
    }
}
