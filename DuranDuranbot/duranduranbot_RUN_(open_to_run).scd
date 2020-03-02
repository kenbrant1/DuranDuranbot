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

As of SuperCollider version 3.10.2 this program has been tested to work.
If this program doesn't work for some reason or has bugs, you can try an older version of SuperCollider,
.. or you can try to fix the issue(s) :) <3

...

DuranDuranbot, and any file packaged with it, is in the public domain.
Anyone is able to use or edit them in any way, for any purpose.

...

contact email: ken_brant@ymail.com or kenbrant7@gmail.com

*/

~ddbot_root = Document.current.dir;

if(~amp == nil, {~amp = 1;});
if(~bit_duration == nil, {~bit_duration_default = 9; ~bit_duration = FileReader.read((~ddbot_root)++"/program_files/other_program_files/bit_speed.scd", false, true).flatten.at(0).asString.asFloat;});
if(~rate_time_range == nil, {~rate_time_range_default = 7; ~rate_time_range = FileReader.read((~ddbot_root)++"/program_files/other_program_files/rate_time_range.scd", false, true).flatten.at(0).asString.asFloat;}); // range in number of seconds to rate a piece from current time backwards. defualt is 8.
if(~rate_button_default == nil, {~rate_button_default = 8;}); // set default point rating at boot. Available point numbers - 1, 2, 4, 8, or 16

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


if(FileReader.read((~ddbot_root)++"/program_files/ddbot_training_data_folder/ddbot_training_data/zdummyfile.scd", false, true) != nil,
	{if(Server.default.serverRunning == false, {s.boot; s.waitForBoot({~duranduranbot_GUI_func.value;});}, {~duranduranbot_GUI_func.value;});},
	{
		w = Window("DuranDuranbot",
			Rect(Window.availableBounds.width/4, Window.availableBounds.height/4, Window.availableBounds.width/2, Window.availableBounds.height/2),
			resizable: true, border:true);

		~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
		/*view.clearOnRefresh = false;*/
		~firstview.background = ~stringcolor;
		~firstviewboundsWIDTH = ~firstview.bounds.width;
		~firstviewboundsHEIGHT = ~firstview.bounds.height;

		a = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/3.4774535809019));
		a.background = ~stringcolor;
		a.string = "DDbot training data files were not detected. Please make sure that the -entire- DuranDuranbot.zip file has been extracted to a location on your computer, including the .zip file in the following location - "++(((~ddbot_root)++"/program_files/ddbot_training_data_folder").quote)++" ... and then open the "++"duranduranbot_RUN_(open_to_rune).scd".quote++" and run it.";
		a.align = \topLeft;
		a.font = Font(size: ~firstviewboundsWIDTH/48);

		w.front;
});

);