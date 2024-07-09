use std::time::Duration;

use bevy::prelude::*;

use crate::gridworld::LevelLoader;

/// Describes and handles logic for various screens.
pub struct ScreensPlayPlugin;

impl Plugin for ScreensPlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_state(ScreenState::TitleScreen)
            .add_event::<MenuButtonPressedEvent>()
            .add_systems(OnEnter(ScreenState::TitleScreen), init_title_screen)
            .add_systems(OnExit(ScreenState::TitleScreen), destroy_title_screen)
            .add_systems(OnEnter(ScreenState::Game), init_game)
            .add_systems(OnExit(ScreenState::Game), destroy_game)
            .add_systems(
                Update,
                (
                    handle_menu_btns_on_change,
                    handle_title_screen_btns,
                    handle_menu_btns,
                ),
            );
    }
}

/// The screens we can be on.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, States)]
pub enum ScreenState {
    TitleScreen,
    Game,
}

/// Denotes the title screen.
#[derive(Component)]
struct TitleScreen;

/// Actions that can be performed on the title screen.
#[derive(Component)]
enum TitleScreenAction {
    Start,
    About,
}

fn init_title_screen(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font_bold = asset_server.load("fonts/montserrat/Montserrat-Bold.ttf");
    let font_regular = asset_server.load("fonts/montserrat/Montserrat-Regular.ttf");
    let btn_text_style = TextStyle {
        font: font_regular.clone(),
        font_size: 22.,
        color: Color::WHITE,
    };

    commands.spawn((Camera2dBundle::default(), IsDefaultUiCamera));
    commands
        .spawn((
            TitleScreen,
            NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    ..default()
                },
                background_color: Color::BLACK.into(),
                ..default()
            },
        ))
        .with_children(|p| {
            // Background
            p.spawn(ImageBundle {
                style: Style {
                    position_type: PositionType::Absolute,
                    margin: UiRect::all(Val::Auto),
                    ..default()
                },
                image: asset_server.load("ui/title_screen/background.png").into(),
                z_index: ZIndex::Local(0),
                ..default()
            });

            // Text elements
            p.spawn(NodeBundle {
                style: Style {
                    width: Val::Percent(100.),
                    height: Val::Percent(100.),
                    flex_direction: FlexDirection::Column,
                    display: Display::Flex,
                    ..default()
                },
                background_color: Color::BLACK.with_a(0.9).into(),
                z_index: ZIndex::Local(1),
                ..default()
            })
            .with_children(|p| {
                // Top text
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Percent(100.),
                        height: Val::Percent(50.),
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::End,
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    p.spawn(TextBundle::from_section(
                        "DEMO:",
                        TextStyle {
                            font: font_bold.clone(),
                            font_size: 40.,
                            color: Color::WHITE,
                        },
                    ));
                    p.spawn(TextBundle::from_section(
                        "PURSUER",
                        TextStyle {
                            font: font_bold.clone(),
                            font_size: 64.,
                            color: Color::WHITE,
                        },
                    ));
                });
                // Options
                p.spawn(NodeBundle {
                    style: Style {
                        width: Val::Px(128.),
                        height: Val::Percent(50.),
                        margin: UiRect::horizontal(Val::Auto),
                        display: Display::Flex,
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        justify_content: JustifyContent::Start,
                        padding: UiRect::vertical(Val::Px(16.)),
                        ..default()
                    },
                    ..default()
                })
                .with_children(|p| {
                    for (action, label) in [
                        (TitleScreenAction::Start, "START"),
                        (TitleScreenAction::About, "ABOUT"),
                    ] {
                        p.spawn((
                            action,
                            MenuButton::default(),
                            ButtonBundle {
                                style: Style {
                                    display: Display::Flex,
                                    width: Val::Percent(100.),
                                    padding: UiRect::axes(Val::Auto, Val::Px(4.)),
                                    border: UiRect::all(Val::Px(1.)),
                                    flex_direction: FlexDirection::Column,
                                    align_items: AlignItems::Center,
                                    ..default()
                                },
                                background_color: Color::WHITE.with_a(0.).into(),
                                ..default()
                            },
                        ))
                        .with_children(|p| {
                            p.spawn(TextBundle::from_section(label, btn_text_style.clone()));
                        });
                    }
                });
            });
        });
}

fn destroy_title_screen(mut commands: Commands, screen_query: Query<Entity, With<TitleScreen>>) {
    let screen_e = screen_query.single();
    commands.entity(screen_e).despawn_recursive();
}

fn handle_title_screen_btns(
    mut ev_btn_pressed: EventReader<MenuButtonPressedEvent>,
    action_query: Query<&TitleScreenAction>,
) {
    for ev in ev_btn_pressed.read() {
        if let Ok(action) = action_query.get(ev.sender) {
            match action {
                TitleScreenAction::Start => info!("Start"),
                TitleScreenAction::About => info!("About"),
            }
        }
    }
}

/// A standard menu button.
#[derive(Default, Component)]
pub struct MenuButton {
    pub hover_start: Duration,
}

/// Sent by buttons when they're pressed.
#[derive(Event)]
pub struct MenuButtonPressedEvent {
    pub sender: Entity,
}

/// Handles menu button states on change.
fn handle_menu_btns_on_change(
    mut btn_query: Query<
        (
            Entity,
            &mut MenuButton,
            &Interaction,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        (Changed<Interaction>, With<Button>),
    >,
    mut ev_btn_pressed: EventWriter<MenuButtonPressedEvent>,
    time: Res<Time>,
) {
    for (e, mut btn, interaction, mut bg_color, mut border_color) in btn_query.iter_mut() {
        match interaction {
            Interaction::Pressed => {
                bg_color.0 = Color::WHITE.with_a(0.);
                border_color.0 = Color::WHITE;
                ev_btn_pressed.send(MenuButtonPressedEvent { sender: e });
            }
            Interaction::Hovered => {
                btn.hover_start = time.elapsed();
                bg_color.0 = Color::WHITE.with_a(0.);
                border_color.0 = Color::WHITE;
            }
            Interaction::None => {
                bg_color.0 = Color::WHITE.with_a(0.);
                border_color.0 = Color::WHITE.with_a(0.);
            }
        }
    }
}

/// Handles menu button states.
fn handle_menu_btns(
    mut btn_query: Query<(&mut MenuButton, &Interaction, &mut BackgroundColor), With<Button>>,
    time: Res<Time>,
) {
    for (btn, interaction, mut bg_color) in btn_query.iter_mut() {
        if interaction == &Interaction::Hovered {
            let elapsed = (time.elapsed() - btn.hover_start).as_secs_f32();
            let amount = (f32::cos(elapsed * std::f32::consts::TAU) + 1.) / 2.;
            bg_color.0 = Color::WHITE.with_a(0.05 * amount);
        }
    }
}

fn init_game(mut commands: Commands) {
    commands.insert_resource(LevelLoader::Path("levels/test.json".into()));
}

fn destroy_game() {}
