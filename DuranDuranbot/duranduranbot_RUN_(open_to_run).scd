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


// ********LOAD SAVED SCORES*********

~loaded_score = {

	// paste score from score file here, within the encapsulating brackets.
	// example -
	/*
	~instrument_1_full_midi = [ 79,79,79,79,79,79,79,79,79,80,80,80,80,80,80,80,80...... etc..... ];
	~instrument_2_full_midi = [ 79,79,79,79,79,79,79,79,79,80,80,80,80,80,80,80,80...... etc..... ];
	~instrument_3_full_midi = [ 79,79,79,79,79,79,79,79,79,80,80,80,80,80,80,80,80...... etc..... ];
	*/
	// Then load DuranDuranbot by pressing "ctrl - enter"/"cmd - return" on your keyboard, and
	// go to the bottom of the "Main Page" and click on the button "Load Saved Score",
	// and click on the "Play" button on top of the Main Page.

	// *Paste score here*



};

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

/*File.readAllString((~ddbot_root)++"/program_files/test_func.scd", r).interpret;
~testfunc.value*/

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


if(FileReader.read((~ddbot_root)++"/program_files/other_program_files/ddbot_training_data_folder/ddbot_training_data/zdummyfile.scd", false, true) != nil,
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
		a.string = "Some program files were not detected. Please make sure that the -entire- DuranDuranbot.zip file has been -fully- extracted to a location on your computer... and then open the "++"duranduranbot_RUN.scd".quote++" from within that extracted DuranDuranbot folder.";
		a.align = \topLeft;
		a.font = Font(size: ~firstviewboundsWIDTH/48);

		w.front;
});



/*

// editor preference settings


(Platform.userConfigDir +/+ "sc_ide_conf.yaml").openDocument;

IDE:
editor:
blinkDuration: 600
disableBlinkingCursor: false
font:
antialias: 2
family: Courier 10 Pitch
size: 9
highlightBracketContents: true
highlightCurrentLine: false
inactiveEditorFadeAlpha: 64
indentWidth: 4
insertMatchingTokens: false
lineWrap: false
spaceIndent: false
stepForwardEvaluation: false
theme: My old theme
themes:
My old theme:
built-in: !textFormat
color: "#6b6bbf"
italic: false
char: !textFormat
color: "#92d38e"
italic: false
class: !textFormat
color: "#6066d2"
italic: false
comment: !textFormat
color: "#bf5454"
italic: false
currentLine: !textFormat
color: "#000000"
background: "#f4f4f4"
italic: false
env-var: !textFormat
color: "#8c654f"
italic: false
evaluatedCode: !textFormat
color: "#000000"
background: "#f8a200"
keyword: !textFormat
color: "#676fe6"
bold: true
italic: false
lineNumbers: !textFormat
color: "#6f6f6f"
background: "#000000"
italic: false
matchingBrackets: !textFormat
color: "#ff0000"
bold: true
mismatchedBrackets: !textFormat
color: "#ffffff"
background: "#960000"
italic: false
number: !textFormat
color: "#9d43a7"
italic: false
postwindowemphasis: !textFormat
color: "#edd400"
bold: true
italic: false
postwindowerror: !textFormat
color: "#d11c24"
italic: false
postwindowsuccess: !textFormat
color: "#738a05"
italic: false
postwindowtext: !textFormat
color: "#ffffff"
italic: false
postwindowwarning: !textFormat
color: "#a57706"
italic: false
primitive: !textFormat
color: "#3333bf"
italic: false
searchResult: !textFormat
color: "#7f7f7f"
background: "#783b23"
selection: !textFormat
{}
string: !textFormat
color: "#bababa"
italic: false
symbol: !textFormat
color: "#87c87e"
italic: false
text: !textFormat
color: "#ffffff"
background: "#000000"
whitespace: !textFormat
color: "#7f7f7f"
italic: false
useComboBox: false
useComboBoxWhenSplitting: false
interpreter:
autoStart: true
configFile: ""
runtimeDir: ""
standalone: false
mainWindow:
detached: !QVariantMap
documents-dock: ""
help-dock: AQHZ0MsAAgAAAAAAdAAAAI8AAAKzAAADBAAAAHQAAACtAAACswAAAwQAAAAAAAAAAAWg
post-dock: ""
geometry: AdnQywACAAAAAAAAAAAAGwAABZ8AAAODAAAAAAAAAC8AAAK/AAAChgAAAAACAAAABaA=
state: AAAA/wAAAAD9AAAAAgAAAAAAAAAAAAAAAPwCAAAAAfsAAAAcAGQAbwBjAHUAbQBlAG4AdABzAC0AZABvAGMAawAAAAAA/////wAAAF8A////AAAAAQAAANYAAAMa/AIAAAAC+wAAABIAaABlAGwAcAAtAGQAbwBjAGsAAAAAFgAAAbUAAAATAP////sAAAASAHAAbwBzAHQALQBkAG8AYwBrAQAAABYAAAMaAAAATQD///8AAATJAAADGgAAAAEAAAACAAAAAQAAAAL8AAAAAA==
postWindow:
lineWrap: false
scrollback: 1000
recentDocuments: !QVariantList
- /home/ken/Documents/duranduranbot/duranduranbot_9000/duranduranbot_9000/duranduranbot_9000_PROGRAM/duranduranbot_RUN.scd
- /home/ken/.config/SuperCollider/sc_ide_conf.yaml
- /home/ken/Downloads/timewavesynth.scd
startWithSession: last


*/

);