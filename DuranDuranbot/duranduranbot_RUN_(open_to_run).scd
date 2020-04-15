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

DuranDuranbot, and any file packaged with it, is in the public domain.
Anyone is able to use or edit them in any way, for any purpose.

...

contact email: ken_brant@ymail.com or kenbrant7@gmail.com

*/

~ddbot_root = Document.current.dir;

if(~tuning_frequency == nil, {~tuning_frequency_default = 440; ~tuning_frequency = FileReader.read((~ddbot_root)++"/program_files/other_program_files/tuning_frequency.scd", false, true).flatten.at(0).asString.asFloat;}); // note A tuneing frequency - default 440 hz
if(~amp == nil, {~amp = 1;});
if(~bit_duration == nil, {~bit_duration_default = 8; ~bit_duration = FileReader.read((~ddbot_root)++"/program_files/other_program_files/bit_speed.scd", false, true).flatten.at(0).asString.asFloat;});
if(~rate_time_range == nil, {~rate_time_range_default = 7; ~rate_time_range = FileReader.read((~ddbot_root)++"/program_files/other_program_files/rate_time_range.scd", false, true).flatten.at(0).asString.asFloat;}); // range in number of seconds to rate a piece from current time backwards. defualt is 8.
if(~net_change_mult == nil, {~rate_button_default = 4; ~net_change_mult = FileReader.read((~ddbot_root)++"/program_files/other_program_files/set_machine_learning_rate.scd", false, true).flatten.at(0).asString.asFloat;}); // note A tuneing frequency - default 440 hz // set default point rating at boot. Available point numbers - 1, 2, 4, 8, or 16

case
{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 1.asString}{~number_of_instruments = 1}
{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 2.asString}{~number_of_instruments = 2}
{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 3.asString}{~number_of_instruments = 3};

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

File.readAllString((~ddbot_root)++"/program_files/duranduranbot_GUI_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/front_page_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/main_page_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/instructions_page_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/player_settings_page_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/save_random_seeds_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/compose_piece_funcs.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/machine_learning_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/synthdefs_func.scd", r).interpret;
File.readAllString((~ddbot_root)++"/program_files/run_piece_func.scd", r).interpret;


if(Server.default.serverRunning == false, {Server.default.boot; Server.default.waitForBoot({~duranduranbot_GUI_func.value;});}, {~duranduranbot_GUI_func.value;});


);