(
/*

Hello!

To run this program... simply click once anywhere on this page,
and press "ctrl - enter"/"cmd - return" on your keyboard.

And if that does not work for some reason, click on the "Language" tab
above, and click on "Evaluate File".

...

This AI music bot is designed to be trained. "Straight out of the box" it
produces random sounds; but you can train it to sound according to
your preferences. More information on how to train DuranDuranbot
and how it learns can be found inside the "click here for instructions" page link
found on the "Main Page" of the program after running it.

...

*If something goes wrong* - you can kill all sounds and processes by hitting "cmd/ctrl - period" on
your keyboard... and if that doesn't work... you can reboot the interpreter
by clicking on the "Language" tab and clicking "Reboot Interpreter", or by highlighting and evaluating (thisProcess.platform.recompile);
and pressing "ctrl - enter"/"cmd - return". Doing this is also beneficial as a restarting or "clean up" method.

...

As of SuperCollider version 3.11.0 this program has been tested to work.
If this program doesn't work for some reason or has bugs, you can try an older version of SuperCollider,
.. or you can try to fix the issue(s) :) <3

...

contact email: ken_brant@ymail.com or kenbrant7@gmail.com
https://github.com/kenbrant1

*/

~ddbot_root = Document.current.dir;

if(~number_of_instruments == nil, {~number_of_instruments = File.readAllString((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", r).interpret;});
if(~tuning_frequency == nil, {~tuning_frequency_default = 440; ~tuning_frequency = FileReader.read((~ddbot_root)++"/program_files/other_program_files/tuning_frequency.scd", false, true).flatten.at(0).asString.asFloat;}); // note A tuneing frequency - default 440 hz
if(~amp == nil, {~amp = 1;});
if(~bit_duration == nil, {~bit_duration_default = 7; ~bit_duration = FileReader.read((~ddbot_root)++"/program_files/other_program_files/bit_speed.scd", false, true).flatten.at(0).asString.asFloat;});
if(~rate_time_range == nil, {~rate_time_range_default = 7; ~rate_time_range = FileReader.read((~ddbot_root)++"/program_files/other_program_files/rate_time_range.scd", false, true).flatten.at(0).asString.asFloat;}); // range in number of seconds to rate a piece from current time backwards. defualt is 8.
if(~net_change_mult == nil, {~rate_button_default = 4; ~net_change_mult = FileReader.read((~ddbot_root)++"/program_files/other_program_files/set_machine_learning_rate.scd", false, true).flatten.at(0).asString.asFloat;}); // note A tuneing frequency - default 440 hz // set default point rating at boot. Available point numbers - 1, 2, 4, 8, or 16
if(~ai_avg_part_length == nil, {~ai_avg_part_length_default = 8; ~ai_avg_part_length = FileReader.read((~ddbot_root)++"/program_files/other_program_files/ai_avg_part_length.scd", false, true).flatten.at(0).asString.asFloat;});
if(~ai_avg_part_range == nil, {~ai_avg_part_range_default = 4; ~ai_avg_part_range = FileReader.read((~ddbot_root)++"/program_files/other_program_files/ai_avg_part_range.scd", false, true).flatten.at(0).asString.asFloat;});
if(~assist_avg_part_length == nil, {~assist_avg_part_length_default = 3; ~assist_avg_part_length = FileReader.read((~ddbot_root)++"/program_files/other_program_files/assist_avg_part_length.scd", false, true).flatten.at(0).asString.asFloat;});
if(~assist_avg_part_range == nil, {~assist_avg_part_range_default = 1; ~assist_avg_part_range = FileReader.read((~ddbot_root)++"/program_files/other_program_files/assist_avg_part_range.scd", false, true).flatten.at(0).asString.asFloat;});
if(~assist_ml_on == nil, {~assist_ml_on = FileReader.read((~ddbot_root)++"/program_files/other_program_files/assist_ml_on.scd", false, true).flatten.at(0).asString.asFloat;});
if(~assist_ml_data_on == nil, {~assist_ml_data_on = FileReader.read((~ddbot_root)++"/program_files/other_program_files/assist_ml_data_on.scd", false, true).flatten.at(0).asString.asFloat;});
if(~fullscreenon == nil, {~fullscreenon = FileReader.read((~ddbot_root)++"/program_files/other_program_files/fullscreenon.scd", false, true).flatten.at(0).asString.asFloat;});
if(~composition_process_setting == nil, {~composition_process_setting = FileReader.read((~ddbot_root)++"/program_files/other_program_files/composition_process_setting.scd", false, true).flatten.at(0).asString.asFloat;});

/*Platform.userExtensionDir;*/

if(FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true) != nil, {
	case
	{FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true).flatten.at(0) == 0.asString}{
		~backgroundcolor = Color.black;
		~stringcolor = Color.white;
	}
	{FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true).flatten.at(0) == 1.asString}{
		~backgroundcolor = Color.white;
		~stringcolor = Color.black;
	};
}, {
	~backgroundcolor = Color.black;
	~stringcolor = Color.white;
});

File.readAllString((~ddbot_root)++"/program_files/main_program_files/part_length_func.scd", r).interpret; "part_length_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/compose_piece_funcs.scd", r).interpret; "compose_piece_funcs.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/duranduranbot_GUI_func.scd", r).interpret; "duranduranbot_GUI_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/main_page_func.scd", r).interpret; "main_page_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/instructions_page_func.scd", r).interpret; "instructions_page_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/player_settings_page_func.scd", r).interpret; "player_settings_page_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/set_seeds_page_func.scd", r).interpret; "set_seeds_page_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/save_random_seeds_func.scd", r).interpret; "save_random_seeds_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/assist_compose_page_func.scd", r).interpret; "assist_compose_page_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/assisted_comp_part_func.scd", r).interpret; "assisted_comp_part_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/assisted_ai_run_piece_func.scd", r).interpret; "assisted_ai_run_piece_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/ai_assist_instructions_page_func.scd", r).interpret; "ai_assist_instructions_page_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/ai_assist_settings_page_func.scd", r).interpret; "ai_assist_settings_page_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/machine_learning_func.scd", r).interpret; "machine_learning_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/synthdefs_func.scd", r).interpret; "synthdefs_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/run_piece_func.scd", r).interpret; "run_piece_func.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/hotkeys.scd", r).interpret; "hotkeys.scd".postln;
File.readAllString((~ddbot_root)++"/program_files/main_program_files/full_midi_to_ml_func.scd", r).interpret; "full_midi_to_ml_func.scd".postln;

if(w != nil, {w.close;});

if(Server.default.serverRunning == false, {Server.default.boot; Server.default.waitForBoot({~duranduranbot_GUI_func.value;});}, {~duranduranbot_GUI_func.value;});


);