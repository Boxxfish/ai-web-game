use bevy::prelude::*;

#[derive(Default)]
pub struct InputPromptPlugin;

impl Plugin for InputPromptPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(InputPromptIconsCfg {
            keyboard: InputTypeIconsCfg {
                path: "input_prompts/keyboard_mouse".into(),
                prefix: "keyboard_".into(),
                suffix: "_outline.png".into(),
            },
        })
        .add_systems(Update, update_prompt);
    }
}

#[derive(Bundle)]
pub struct InputPromptBundle {
    pub input_prompt: InputPrompt,
    pub node_bundle: NodeBundle,
}

impl Default for InputPromptBundle {
    fn default() -> Self {
        Self {
            input_prompt: InputPrompt {
                label: "".into(),
                input: InputType::A,
            },
            node_bundle: default(),
        }
    }
}

/// Denotes an input prompt.
#[derive(Component)]
pub struct InputPrompt {
    pub label: String,
    pub input: InputType,
}

/// The input to use to trigger a command.
#[allow(dead_code)]
pub enum InputType {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Key0,
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    Space,
    Enter,
    LeftArrow,
    RightArrow,
    UpArrow,
    DownArrow,
    ArrowKeys,
    WASD,
}

impl InputType {
    pub fn as_str(&self) -> &str {
        match self {
            InputType::A => "a",
            InputType::B => "b",
            InputType::C => "c",
            InputType::D => "d",
            InputType::E => "e",
            InputType::F => "f",
            InputType::G => "g",
            InputType::H => "h",
            InputType::I => "i",
            InputType::J => "j",
            InputType::K => "k",
            InputType::L => "l",
            InputType::M => "m",
            InputType::N => "n",
            InputType::O => "o",
            InputType::P => "p",
            InputType::Q => "q",
            InputType::R => "r",
            InputType::S => "s",
            InputType::T => "t",
            InputType::U => "u",
            InputType::V => "v",
            InputType::W => "w",
            InputType::X => "x",
            InputType::Y => "y",
            InputType::Z => "z",
            InputType::Key0 => "0",
            InputType::Key1 => "1",
            InputType::Key2 => "2",
            InputType::Key3 => "3",
            InputType::Key4 => "4",
            InputType::Key5 => "5",
            InputType::Key6 => "6",
            InputType::Key7 => "7",
            InputType::Key8 => "8",
            InputType::Key9 => "9",
            InputType::Space => "space",
            InputType::Enter => "enter",
            InputType::LeftArrow => "arrow_left",
            InputType::RightArrow => "arrow_right",
            InputType::UpArrow => "arrow_up",
            InputType::DownArrow => "arrow_down",
            InputType::ArrowKeys => "arrows",
            InputType::WASD => "wasd",
        }
    }
}

/// Describes how icons for an input type are loaded.
#[derive(Default)]
pub struct InputTypeIconsCfg {
    pub path: String,
    pub prefix: String,
    pub suffix: String,
}

/// Indicates how input prompt icons are loaded.
#[derive(Resource)]
pub struct InputPromptIconsCfg {
    pub keyboard: InputTypeIconsCfg,
}

/// Updates prompt when label or input changes.
fn update_prompt(
    mut commands: Commands,
    prompt_query: Query<(Entity, &InputPrompt, Option<&Children>), Changed<InputPrompt>>,
    prompt_cfg: Res<InputPromptIconsCfg>,
    asset_server: Res<AssetServer>,
) {
    for (e, prompt, children) in prompt_query.iter() {
        if let Some(children) = children {
            for child in children.iter() {
                commands.entity(*child).despawn_recursive();
            }
        }
        commands.entity(e).with_children(|p| {
            p.spawn(NodeBundle {
                style: Style {
                    display: Display::Flex,
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.),
                    ..default()
                },
                ..default()
            })
            .with_children(|p| {
                p.spawn(ImageBundle {
                    image: UiImage::new(asset_server.load(format!(
                        "{}/{}{}{}",
                        prompt_cfg.keyboard.path,
                        prompt_cfg.keyboard.prefix,
                        prompt.input.as_str(),
                        prompt_cfg.keyboard.suffix
                    ))),
                    ..default()
                });
                p.spawn(TextBundle::from_section(
                    prompt.label.clone(),
                    TextStyle {
                        font: asset_server.load("fonts/montserrat/Montserrat-Regular.ttf"),
                        font_size: 16.,
                        color: Color::WHITE,
                    },
                ));
            });
        });
    }
}
