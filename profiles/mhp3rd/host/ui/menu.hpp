#pragma once

// The in-game menu as the launcher opens it, before the game has started:
// the same pages, drawn one frame at a time over the launcher's background.
namespace mhp3rd::ui {

// The menu's sections, in the order of its tabs.
enum class MenuTab { Video = 0, Audio, Controls, Network, Mods, System };

// A row the menu can open on instead of its page's first.
enum class MenuFocus { None, TexturePack, Saves };

// Opens the menu at `tab`, on `focus` when given.
void open_launcher_menu(MenuTab tab, MenuFocus focus = MenuFocus::None);
[[nodiscard]] bool launcher_menu_open();
// One frame of the open menu; false once it has closed.
bool launcher_menu_frame();
// Once, after the menu closed because the player quit, chose to set up the game
// data again or to restart (install::setup_requested_on_exit and
// restart_requested_on_exit tell which).
[[nodiscard]] bool take_launcher_menu_quit();

} // namespace mhp3rd::ui
