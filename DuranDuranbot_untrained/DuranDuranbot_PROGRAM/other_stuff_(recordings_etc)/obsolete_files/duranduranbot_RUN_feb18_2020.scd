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

	// Copy and paste the all lines below in the "Load Saved Score" section in the top of the "duranduranbot_RUN.scd" file
	~instrument_1_full_midi = [ 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 69, 69, 69, 69, 69, 69, 69, 69, 69, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 64, 63, 63, 63, 63, 63, 63, 63, 63, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 73, 73, 73, 73, 73, 73, 73, 73, 73, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 64, 64, 64, 64, 64, 64, 64, 64, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 73, 73, 73, 73, 73, 73, 73, 73, 73, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75 ];
	~instrument_2_full_midi = [ 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 71, 71, 71, 71, 71, 66, 66, 62, 62, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 62, 62, 62, 62, 62, 62, 62, 62, 64, 64, 64, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 64, 64, 64, 64, 64, 64, 64, 64, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60 ];





};


// Things to do -
// MAKE SURE IT WORKS - Test ENTIRE machine learning process. Record ALL rated stuff, see if they
// are saved in data, then see if it works rest of process.
//
// Create reverse rate. Undo rate.
//
// Put all training data into one file.
//
// think about how to add rests into piece
//
// think about getting rid of "players", just have 1 player. And having the option to combine the 1 player of each DDbot
// to form a "band".
//
// Do this - put different players data in different folders, so you can mix and match players.
//
// Combine different composition techniques for AI... combine machine learning with variational method... compose section
// of a certain length, pattern, etc... then do variation of that in next section. For instance keep length, but vary pattern...
// use machine learning for general tendecies in note/time choices, etc... MODULES... have various micro-modules under a macro-module...
// heirarchical... use as little space for training data as possible.
//
// FIGURED IT OUT - Start with one "base" chord. Run composition process for certain amount of time. Then next "base" chord.
// Run composition process for same amount of time....... Hierarchical compoisition. Have "base" chords with composition
// in between. Use same "base" chord progression over and over again. Number of base chords could be as many as you want.
// Maybe even have base chords for the base chord progression.
//
// Have DDbot machine learn preferences in that^ (meaning in the structure of heirarhy, etc)
//
// HAVE DDBOT THINK DURAING COMPOSITION! - Use ML data to make 1st and last note of a "verse". Create rest of verse
// by using ML data from both first and last note, meet in middle, "compromise" in middle, then "compromise" based on that
// back to the first and last notes. This is the AI REALLY composing! THINKING!
// STILL use rest of hieriarchil structure. Notes will change, but structure will stay the same. Notes will only change
// on the "bottom" level.
//
// Or RATHER - Have beginning of first and last note create their own "sequence" for the length of the verse,
// then "blend" the sequences by putting overlapping them on top of each other... find middle ground
// of each note... bit by bit.
//
// When "blending" two notes, use machine learning in case it results in decimal. Actually, perhaps use ML for note difference
// instead... like - if note difference is *this* (within 12 note range) -... actually probably don't do that. There is no correlation.
//
// Create 1st chord sequence and 2nd chord sequence, blend, then use same 2nd chord sequence for 2nd and 3rd chord
// sequence blend.
//
// HOW TO OVERLAP SEQUENCES - Example: Create 1st and 2nd sequences, overlap them, "blend" EVERY NOTE - including
// "original" starting chords... then create NEW sequence for the 2nd chord, and do the same.
//
// DO ML on length of sequences too
//
// Do overlap... then use the overlap result in next sequence, overlapping that with next generated sequence.


// KEY TO SUCCESSFUL AI - Allow for most machine learning in the least amount of time. ML in lots of different things.
// (ex - not just note transitions, etc..... but also *literally* everything else.
// Again - Make the most machine learning in the least amount of time.

// Current bugs - every time go to main page while piece is playing, the piece goes to it's beginning for a slight moment then goes back to current position.

////////////////////////////////////////////////////////////////////////////////////////

// GUI

if(~amp == nil, {~amp = 1;});
if(~bit_duration == nil, {~bit_duration_default = 7; ~bit_duration = FileReader.read(Document.current.dir++"/other_program_files/bit_speed.scd", false, true).flatten.at(0).asString.asFloat;});
if(~auto_select_diff == nil, {~auto_select_diff = 7;}); // range in number of seconds to rate a piece from current time backwards. defualt is 7.
if(~rate_button_default == nil, {~rate_button_default = 8;}); // set default point rating at boot. Available point numbers - 1, 2, 4, 8, or 16

case
{FileReader.read(Document.current.dir++"/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 2.asString}{~number_of_instruments = 2}
{FileReader.read(Document.current.dir++"/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 3.asString}{~number_of_instruments = 3};

if(FileReader.read(Document.current.dir++"/other_program_files/gui_color.scd", false, true) != nil, {
	case
	{FileReader.read(Document.current.dir++"/other_program_files/gui_color.scd", false, true).flatten.at(0) == 0.asString}{
		~backgroundcolor = Color.black;
		~stringcolor = Color.white;
	}
	{FileReader.read(Document.current.dir++"/other_program_files/gui_color.scd", false, true).flatten.at(0) == 1.asString}{
		~backgroundcolor = Color.white;
		~stringcolor = Color.black;
	};
}, {
	~backgroundcolor = Color.black;
	~stringcolor = Color.white;
});

~duranduranbot_GUI_func = {

	var w, /*~firstview,*/ firstviewtext1, firstviewtext2, firstviewtext3, firstviewtext4, firstviewtext5, firstviewtext6, firstviewnextbutton, mainview, mainviewtext, randseedEZText1, randseedEZText2, randseedEZText3, randseed_1_make, randseed_2_make, randseed_3_make, randseedviewtext1,randseedviewtext2,
	randseedviewtext3, numerologynameEZText1, numerologynameEZText2, numerologynameEZText3, numerologynameviewtext1, numerologynameviewtext2, numerologynameviewtext3,
	generatepiecebutton, playbutton, enterdatabasebutton, recordbutton, songtitletext, songtitlestatictext, piecedurationstatictext,piecepositionstatictext,looptextfield_1,looptextfield_2,loopsetbutton,stoploopbutton,looptextfield_3,newpieceautobutton,newpieceautotext, machinelearningbutton, machinelearningtext,loadscorebuttonbutton,loadscoretext,
	onestarbutton,twostarbutton,threestarbutton,fourstarbutton,fivestarbutton, quicknewpiecebutton, quicknewpiecebutton_revert, quicknewpiecewait,
	rate_button_1,rate_button_2,rate_button_3,rate_button_4,rate_button_5,rate_button_6, rate_button,rate_revert_button,ratepiecetext,ratedonetext, /*rangeslider,*/timeslider,timeslidertext,
	rate_selectionstatictext,ctrlperiodtext,training_settings_page_button,playerchoicetext,save_current_piece_button,playerspeedtextfield,playerspeedtextfield2,
	closeddbotbutton;
	w = Window("DuranDuranbot",
		Rect(Window.availableBounds.width*(1/4.4), Window.availableBounds.height*(1/4.5), Window.availableBounds.width*((4.5-2)/4.5), Window.availableBounds.height*((4.5-2)/4.5)),
		resizable: true, border:true);

	~front_page_func = {

		~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
		/*view.clearOnRefresh = false;*/
		~firstview.background = ~backgroundcolor;
		~firstviewboundsWIDTH = ~firstview.bounds.width;
		~firstviewboundsHEIGHT = ~firstview.bounds.height;


		firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/2.9774535809019));
		firstviewtext1.background=~backgroundcolor;
		firstviewtext1.stringColor_(~stringcolor);
		firstviewtext1.string = "Hello. This is DuranDuranbot. \n\nTo enhance the creativity of this AI bot... please consider providing it with some personal creative inspiration of yours by entering 3 number values into the 3 fields below, numerical values up to 8 digits long, and hit Enter. Consider writing down the 3 values you gave it to be able to replicate exactly the music made by this AI in this specific instance at a later date - although they will still be saved in the "++"duranduranbot_saved_random _seeds.scd".quote++" file, accessible on the next page.";
		firstviewtext1.align = \topLeft;
		firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);


		randseedEZText1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.6, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
		randseedEZText1.string = "Random Seed 1";
		randseedEZText1.background_(~backgroundcolor);
		randseedEZText1.stringColor_(~stringcolor);
		randseedEZText1.font = Font(size: ~firstviewboundsWIDTH/48);
		randseedEZText1.action = {
			randseed_1_make = randseedEZText1.value;
			randseed_1_make = randseed_1_make.asString++"";
			randseed_1_make = randseed_1_make.separate;
			if(randseed_1_make.size > 8, {randseed_1_make = nil; randseedEZText1.string = "must be 8 digits or less";}, {
				randseed_1_make = randseed_1_make.join.asString.interpret;
				randseedviewtext1.string = if(randseed_1_make.isNumber == true, {
					randseed_1_make = randseed_1_make.asFloat; randseed_1_make.postln; ~randseed_1 = randseed_1_make; ~save_random_seeds_func.value; ~randseed_1.asString++" entered";}, {
					randseed_1_make = nil; randseedEZText1.string = "Enter Numerical Value";
				});
			});
		};

		randseedEZText2 = TextField(~firstview, Rect(~firstviewboundsWIDTH/3.7894736842105, ~firstviewboundsHEIGHT/2.6, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
		randseedEZText2.string = "Random Seed 2";
		randseedEZText2.background_(~backgroundcolor);
		randseedEZText2.stringColor_(~stringcolor);
		randseedEZText2.font = Font(size: ~firstviewboundsWIDTH/48);
		randseedEZText2.action = {
			randseed_2_make = randseedEZText2.value;
			randseed_2_make = randseed_2_make.asString++"";
			randseed_2_make = randseed_2_make.separate;
			if(randseed_2_make.size > 8, {randseed_2_make = nil; randseedEZText2.string = "must be 8 digits or less";}, {
				randseed_2_make = randseed_2_make.join.asString.interpret;
				randseedviewtext2.string = if(randseed_2_make.isNumber == true, {
					randseed_2_make = randseed_2_make.asFloat; randseed_2_make.postln; ~randseed_2 = randseed_2_make; ~save_random_seeds_func.value; ~randseed_2.asString++" entered";}, {
					randseed_2_make = nil; randseedEZText2.string = "Enter Numerical Value";
				});
			});
		};

		randseedEZText3 = TextField(~firstview, Rect(~firstviewboundsWIDTH/1.8947368421053, ~firstviewboundsHEIGHT/2.6, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
		randseedEZText3.string = "Random Seed 3";
		randseedEZText3.background_(~backgroundcolor);
		randseedEZText3.stringColor_(~stringcolor);
		randseedEZText3.font = Font(size: ~firstviewboundsWIDTH/48);
		randseedEZText3.action = {
			randseed_3_make = randseedEZText3.value;
			randseed_3_make = randseed_3_make.asString++"";
			randseed_3_make = randseed_3_make.separate;
			if(randseed_3_make.size > 8, {randseed_3_make = nil; randseedEZText3.string = "must be 8 digits or less";}, {
				randseed_3_make = randseed_3_make.join.asString.interpret;
				randseedviewtext3.string = if(randseed_3_make.isNumber == true, {
					randseed_3_make = randseed_3_make.asFloat; randseed_3_make.postln; ~randseed_3 = randseed_3_make; ~save_random_seeds_func.value; ~randseed_3.asString++" entered";}, {
					randseed_3_make = nil; randseedEZText3.string = "Enter Numerical Value";
				});
			});
		};

		randseedviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.26, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
		randseedviewtext1.background=~backgroundcolor;
		randseedviewtext1.stringColor_(~stringcolor);
		randseedviewtext1.string = if(~randseed_1 != nil, {~randseed_1.asString++" entered"}, {"number not entered"});
		randseedviewtext1.align = \topLeft;
		randseedviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

		randseedviewtext2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.7894736842105,~firstviewboundsHEIGHT/2.26, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
		randseedviewtext2.background=~backgroundcolor;
		randseedviewtext2.stringColor_(~stringcolor);
		randseedviewtext2.string = if(~randseed_2 != nil, {~randseed_2.asString++" entered"}, {"number not entered"});
		randseedviewtext2.align = \topLeft;
		randseedviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

		randseedviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.8947368421053, ~firstviewboundsHEIGHT/2.26, ~firstviewboundsWIDTH/4.2352941176471, ~firstviewboundsHEIGHT/21));
		randseedviewtext3.background=~backgroundcolor;
		randseedviewtext3.stringColor_(~stringcolor);
		randseedviewtext3.string = if(~randseed_3 != nil, {~randseed_3.asString++" entered"}, {"number not entered"});
		randseedviewtext3.align = \topLeft;
		randseedviewtext3.font = Font(size: ~firstviewboundsWIDTH/48);


		firstviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.415725806452, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/10.85));
		firstviewtext3.background=~backgroundcolor;
		firstviewtext3.stringColor_(~stringcolor);
		firstviewtext3.string = "Or you can do none of that, and autogenerate all of those things by clicking on "++"Next".quote++" below... Which you have to do regardless :)...";
		firstviewtext3.align = \topLeft;
		firstviewtext3.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewnextbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/1.2116451016636, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/10.925)).states_([["Next",~stringcolor,~backgroundcolor],["Next",~stringcolor,~backgroundcolor]]).action_({arg button;

			if(~randseed_1 == nil, {~randseed_1 =  99999999.rand;});
			if(~randseed_2 == nil, {~randseed_2 =  99999999.rand;});
			if(~randseed_3 == nil, {~randseed_3 =  99999999.rand;});

			if(~startprogram == nil, {~save_random_seeds_func.value; ~startprogram = 1;});

			~main_page_func.value;
		});
		firstviewnextbutton.font = Font(size: ~firstviewboundsWIDTH/48);

	};

	~main_page_func = {var arrow, newrandomseedsbutton, reversenewrandomseedsbutton, currentrandomseedstrext, playnil, see_saved_seeds_file_button;

		~firstview.close;

		~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
		/*view.clearOnRefresh = false;*/
		~firstview.background = ~backgroundcolor;
		~firstviewboundsWIDTH = ~firstview.bounds.width;
		~firstviewboundsHEIGHT = ~firstview.bounds.height;

		firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21));
		firstviewtext1.background=~backgroundcolor;
		firstviewtext1.stringColor_(~stringcolor);
		firstviewtext1.string = "Main Page";
		firstviewtext1.align = \topLeft;
		firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["Back",~stringcolor,~backgroundcolor],["Back",~stringcolor,~backgroundcolor]]).action_({arg button;
			~firstview.close;
			~front_page_func.value;
		});

		firstviewtext2.string = "Back";
		firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewtext3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.3584905660377, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Click Here For Instructions",~stringcolor,~backgroundcolor],["Click Here For Instructions",~stringcolor,~backgroundcolor]]).action_({arg button;
			~instructions_page_func.value;

		});
		firstviewtext3.font = Font(size: ~firstviewboundsWIDTH/48);

		generatepiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/6.5454545454545, ~firstviewboundsHEIGHT/21)).states_([["Generate Piece",~stringcolor,~backgroundcolor],["Generate Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			var piecegenerated;
			~piecegenerated = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695652174, ~firstviewboundsWIDTH/36));
			~piecegenerated.background=~backgroundcolor;
			~piecegenerated.stringColor_(~stringcolor);
			"Please Wait".postln;
			~piecegenerated.font = Font(size: ~firstviewboundsWIDTH/55);
			~piecegenerated.string = "Please Wait";/*
			if(~piece_created == "true", {~playbutton.close; ~arrow.close;});*/
			~piece_generator_func.value;

		});
		generatepiecebutton.font = Font(size: ~firstviewboundsWIDTH/48);

		/*if(playnil == nil, {nil}, {*/
		/*~play_visible_func = {
		~play_visible_func_verify*/

		~playbutton_func = {
			~playbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/4.95, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Play Piece",~stringcolor,~backgroundcolor],["Pause Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
				var pause_piece;
				if(button.value == 1, {
					if(~pause_piece != 1, {
						~seedscheck = 1;
						~stopbuttonfunc.value;
						~run_piece_func.value;
					}, {/*~timer_routine.resume;*/
						~amp = 1;
						Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, ~amp);});});
						~duranduranbot_routine.resume; "Piece Resumed".postln; if(~will_record == 1, {s.record;});});
					~pause_piece = 0;
				}, {
					/*~timer_routine.pause;*/
					Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, 0);});});

					~duranduranbot_routine.pause;
					/*Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.set(\freq, 0);});});
					*/"Piece Paused".postln;
					if(~will_record == 1, {s.pauseRecording;});
					~pause_piece = 1;
					~playbutton.string = "Resume";
				});

			});
			~playbutton.font = Font(size: ~firstviewboundsWIDTH/48);
			/*};*/
			~arrow = StaticText(~firstview, Rect(~firstviewboundsWIDTH/5.7016393442623, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/21));
			~arrow.background=~backgroundcolor;
			~arrow.stringColor_(~stringcolor);
			~arrow.string = "---";
			~arrow.font = Font(size: ~firstviewboundsWIDTH/48);
		};

		~playbutton_func.value;
		if(~pause_piece == 1, {~playbutton.string = "Resume"; ~stopbuttonfunc.value;});

		if(~piece_created == nil, {~playbutton.close; ~arrow.close;});

		~stop_func = {
			AppClock.sched(0, {
				/*~timer_routine.stop;*/
				/*Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(\amp, 0);});});
				*/	/*~piece_position = 0;*/
				timeslider.valueAction = 0;
				"piece stopped".postln;
				~fullpiecedur2 = ~fullpiecedur;
				~stopbutton.close;
				~playbutton.value = 0;
				~pause_piece = 0;
				~playbutton.string = "Play Piece";
				Server.default.makeBundle(Server.default.latency, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.free;});});
				~duranduranbot_routine.stop;
				if(~will_record == 1, {s.stopRecording;});
				Server.freeAll;
			});
		};

		~stopbuttonfunc = {~stopbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.93, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Stop piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			~stop_func.value;
		});

		~stopbutton.font = Font(size: ~firstviewboundsWIDTH/48);
		};

		if(~duranduranbot_routine.isPlaying == true, {~playbutton.value = 1; ~stopbuttonfunc.value}, {~playbutton.value = 0});

		~net_change_mult_array = [1, 2, 4, 8, 16];

		rate_button_1 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/5.29, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x1 pts",~stringcolor,~backgroundcolor], ["x1 pts",~stringcolor,Color.blue]]).action_({arg button;
			~net_change_mult = ~net_change_mult_array.at(0);
			rate_button_1.value = 1;
			rate_button_2.value = 0;
			rate_button_3.value = 0;
			rate_button_4.value = 0;
			rate_button_5.value = 0;
		});
		rate_button_1.font = Font(size: ~firstviewboundsWIDTH/48);

		rate_button_2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/3.62, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x2 pts",~stringcolor,~backgroundcolor], ["x2 pts",~stringcolor,Color.blue]]).action_({arg button;
			~net_change_mult = ~net_change_mult_array.at(1);
			rate_button_1.value = 0;
			rate_button_2.value = 1;
			rate_button_3.value = 0;
			rate_button_4.value = 0;
			rate_button_5.value = 0;
		});
		rate_button_2.font = Font(size: ~firstviewboundsWIDTH/48);

		rate_button_3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.75, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x4 pts",~stringcolor,~backgroundcolor], ["x4 pts",~stringcolor,Color.blue]]).action_({arg button;
			~net_change_mult = ~net_change_mult_array.at(2);
			rate_button_1.value = 0;
			rate_button_2.value = 0;
			rate_button_3.value = 1;
			rate_button_4.value = 0;
			rate_button_5.value = 0;
		});
		rate_button_3.font = Font(size: ~firstviewboundsWIDTH/48);

		rate_button_4 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.22, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x8 pts",~stringcolor,~backgroundcolor], ["x8 pts",~stringcolor,Color.blue]]).action_({arg button;
			~net_change_mult = ~net_change_mult_array.at(3);
			rate_button_1.value = 0;
			rate_button_2.value = 0;
			rate_button_3.value = 0;
			rate_button_4.value = 1;
			rate_button_5.value = 0;
		});
		rate_button_4.font = Font(size: ~firstviewboundsWIDTH/48);

		rate_button_5 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.855, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["x16 pts",~stringcolor,~backgroundcolor], ["x16 pts",~stringcolor,Color.blue]]).action_({arg button;
			~net_change_mult = ~net_change_mult_array.at(4);
			rate_button_1.value = 0;
			rate_button_2.value = 0;
			rate_button_3.value = 0;
			rate_button_4.value = 0;
			rate_button_5.value = 1;
		});
		rate_button_5.font = Font(size: ~firstviewboundsWIDTH/48);

		rate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Rate",~backgroundcolor,Color.green]]).action_({arg button;

			if(~duranduranbot_routine.isPlaying == true, {
				q = FileReader.read(Document.current.dir++"/other_program_files/ddbot_machine_learning_on_off.scd");
				2.do({q = q.at(0)});
				q = q.asInteger;
				if(q == 1, {~machine_learning_func.value;});
				("\nlast "++(~auto_select_diff)++" secs rated:\nx"++(~net_change_mult)++" points\n").postln;
			}, {
				"pice must be running to rate".postln;
				if(ratedonetext != nil, {ratedonetext.close; ratedonetext = nil;});
				ratedonetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.063829787234, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/3.3, ~firstviewboundsHEIGHT/21));
				ratedonetext.background=~backgroundcolor;
				ratedonetext.stringColor_(~stringcolor);
				ratedonetext.string = "PIECE MUST BE RUNNING TO RATE";
				ratedonetext.font = Font(size: ~firstviewboundsWIDTH/55);

				AppClock.sched(3, {if(ratedonetext != nil, {ratedonetext.close; ratedonetext = nil;});});
			});
		});
		rate_button.font = Font(size: ~firstviewboundsWIDTH/48);

		if(~net_change_mult == nil, {
			case
			{~rate_button_default == ~net_change_mult_array.at(0);}{rate_button_1.valueAction = 0;}
			{~rate_button_default == ~net_change_mult_array.at(1);}{rate_button_2.valueAction = 0;}
			{~rate_button_default == ~net_change_mult_array.at(2);}{rate_button_3.valueAction = 0;}
			{~rate_button_default == ~net_change_mult_array.at(3);}{rate_button_4.valueAction = 0;}
			{~rate_button_default == ~net_change_mult_array.at(4);}{rate_button_5.valueAction = 0;}
		}, {
			case
			{~net_change_mult == ~net_change_mult_array.at(0);}{rate_button_1.value = 1;}
			{~net_change_mult == ~net_change_mult_array.at(1);}{rate_button_2.value = 1;}
			{~net_change_mult == ~net_change_mult_array.at(2);}{rate_button_3.value = 1;}
			{~net_change_mult == ~net_change_mult_array.at(3);}{rate_button_4.value = 1;}
			{~net_change_mult == ~net_change_mult_array.at(4);}{rate_button_5.value = 1;};
		});

		/*rate_revert_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/9.88, ~firstviewboundsHEIGHT/3.228571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Revert Rate",~stringcolor,~backgroundcolor]]).action_({arg button;
		~ml_revert_func.value;
		});
		rate_revert_button.font = Font(size: ~firstviewboundsWIDTH/68);*/

		ratepiecetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.67571428571, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/12));
		ratepiecetext.background=~backgroundcolor;
		ratepiecetext.stringColor_(~stringcolor);
		ratepiecetext.string = "^rate last "++(~auto_select_diff)++" seconds from current position in piece by clicking on a xpt rating above, and clicking "++"Rate".quote++". Higher point # = faster training, but less fine-tuning.";
		ratepiecetext.font = Font(size: ~firstviewboundsWIDTH/55);

		/*rangeslider = EZRanger(~firstview, bounds: (Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.1951428571, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"), action: {
		|rangeselect|
		var rate_selection_start1;
		~rate_selection_start = (rangeselect.value.at(0)*(~instrument_1_full.size/~piecedur_secs));
		rate_selection_start1 = ~rate_selection_start.roundUp;
		if(rate_selection_start1 != ~rate_selection_start, {~rate_selection_start = rate_selection_start1-1});
		~rate_selection_end = (rangeselect.value.at(1)*(~instrument_1_full.size/~piecedur_secs)).roundUp;
		/*rangeselect.value.postln;*/

		}, initVal: [0,if(~piecedur_secs != nil, {~piecedur_secs}, {0})]);
		rangeslider.setColors(~stringcolor,~stringcolor, ~stringcolor(0.7),~stringcolor, ~stringcolor, Color.yellow);
		rangeslider.font = Font(size: ~firstviewboundsWIDTH/55);*/

		/*rate_selectionstatictext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.502222222222 /*~firstviewboundsHEIGHT/6.2428571428571*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
		rate_selectionstatictext.background=~stringcolor;
		rate_selectionstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"piece length - --:-- mins"});
		rate_selectionstatictext.font = Font(size: ~firstviewboundsWIDTH/48);*/

		if(~piece_position == nil, {~piece_position = 0});

		timeslider = EZSlider(~firstview,  bounds: (Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.073, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"), action: {
			|timeselect|
			var selection1;
			~piece_position = timeselect.value*(~instrument_1_full.size/~piecedur_secs); // works
			selection1 = ~piece_position.roundUp;
			if(selection1 != ~piece_position, {~piece_position = selection1-1});
			/*if(timeslider.value >= ~auto_select_diff, {rangeslider.valueAction = [timeslider.value-~auto_select_diff, timeslider.value]}, {rangeslider.valueAction = [0, timeslider.value]});
			*//*timeselect.value.postln;
			~piece_position.postln;*/
			if((~piecedur_secs_zero+1) != timeselect.value, {
				~piecedur_secs_zero = timeselect.value;
				~fullpiecedur_change = ((~piecedur_secs - ~piecedur_secs_zero)/60);
				~dd_secs = ((~fullpiecedur_change-(~dd_mins = ~fullpiecedur_change.roundUp-1))*60).roundUp;
			});

		});
		timeslider.setColors(~stringcolor,~stringcolor);

		if(~pause_piece != nil, {if(~pause_piece != 1,timeslider.valueAction = 0; {/*rangeslider.valueAction = [0, if(~piecedur_secs != nil, {~piecedur_secs}, {0});]*/nil});});

		/*if(~pause_piece == 0, {~piece_position = 0});*/

		timeslidertext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.84, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/22.55));
		timeslidertext.background=~backgroundcolor;
		timeslidertext.stringColor=~stringcolor;
		timeslidertext.string = "^Jump to location in current piece by using the slider just above";
		timeslidertext.font = Font(size: ~firstviewboundsWIDTH/55);


		quicknewpiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/6.1454545454545, ~firstviewboundsHEIGHT/21)).states_([["Quick New Piece",~stringcolor,~backgroundcolor],["Quick New Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			var quicknewpiecewait;
			quicknewpiecewait = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695, ~firstviewboundsHEIGHT/22.55));
			quicknewpiecewait.background_(~backgroundcolor);
			quicknewpiecewait.stringColor_(~stringcolor);
			quicknewpiecewait.string = "please wait";
			quicknewpiecewait.font = Font(size: ~firstviewboundsWIDTH/55);
			"please wait".postln;
			if(~piecegenerated != nil, {~piecegenerated.close});
			~quicknewpiecebutton = 1;
			{if(~duranduranbot_routine.isPlaying == true, {~stop_func.value});}.value;
			AppClock.sched(0, {
				if(~seedscheck == 1, {newrandomseedsbutton.valueAction = 0}, {~seedscheck = 1});
				generatepiecebutton.valueAction = 0; //piece fires at end of piece gen func
			});
			quicknewpiecewait.close;
		});
		quicknewpiecebutton.font = Font(size: ~firstviewboundsWIDTH/48);

		quicknewpiecebutton_revert = Button.new(~firstview, Rect(~firstviewboundsWIDTH/5.4, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/7.9454545454545, ~firstviewboundsHEIGHT/21)).states_([["Revert Piece",~stringcolor,~backgroundcolor],["Revert Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			var quicknewpiecewait;
			quicknewpiecewait = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/9.5, ~firstviewboundsWIDTH/6.2608695, ~firstviewboundsHEIGHT/22.55));
			quicknewpiecewait.background_(~stringcolor);
			quicknewpiecewait.stringColor_(~stringcolor);
			quicknewpiecewait.string = "Please Wait";
			if(~piecegenerated != nil, {~piecegenerated.close});
			quicknewpiecewait.font = Font(size: ~firstviewboundsWIDTH/55);
			"please wait".postln;
			~quicknewpiecebutton = 1;

			if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});

			AppClock.sched(0, {
				if(~seedscheck == 1, {reversenewrandomseedsbutton.valueAction = 0}, {~seedscheck = 1});
				generatepiecebutton.valueAction = 0; //piece fires at end of piece gen func
			});
			quicknewpiecewait.close;
		});
		quicknewpiecebutton_revert.font = Font(size: ~firstviewboundsWIDTH/48);

		recordbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.35, ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["Record Off",~stringcolor,~backgroundcolor],["Record On",~backgroundcolor,Color.red]]).action_({arg button;
			if(button.value == 1, {
				~recordtext.string = "All pieces WILL be recorded";
				~will_record = 1;
			}, {~recordtext.string = "All pieces will NOT be recorded"; ~will_record = 0;});

		});
		recordbutton.font = Font(size: ~firstviewboundsWIDTH/48);

		~recordtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/7.6, ~firstviewboundsHEIGHT/1.35, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
		~recordtext.background=~backgroundcolor;
		~recordtext.stringColor=~stringcolor;
		~recordtext.string = "All pieces will NOT be recorded";
		~recordtext.font = Font(size: ~firstviewboundsWIDTH/55);

		if(~will_record == 1, {~recordtext.string = "All pieces WILL be recorded"; recordbutton.value = 1;},
			{~recordtext.string = "All pieces will NOT be recorded"; recordbutton.value = 0;});

		piecedurationstatictext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.65 /*~firstviewboundsHEIGHT/6.2428571428571*/, ~firstviewboundsWIDTH/2.95, ~firstviewboundsHEIGHT/21));
		piecedurationstatictext.background=~backgroundcolor;
		piecedurationstatictext.stringColor=~stringcolor;
		piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"Piece Length - --:-- mins"});
		piecedurationstatictext.font = Font(size: ~firstviewboundsWIDTH/48);

		piecepositionstatictext = StaticText(~firstview, Rect((~firstviewboundsWIDTH/72)+(~firstviewboundsWIDTH/2.95)+4, ~firstviewboundsHEIGHT/1.65 /*~firstviewboundsHEIGHT/6.2428571428571*/, ~firstviewboundsWIDTH/2.7, ~firstviewboundsHEIGHT/21));
		piecepositionstatictext.background=~backgroundcolor;
		piecepositionstatictext.stringColor=~stringcolor;
		piecepositionstatictext.string = if(~instrument_1_full_midi != nil, {(
			"Bit pos. "++(~piece_position.asInteger.asString)++" Midi "++(
				case
				{~number_of_instruments == 2}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)}
				{~number_of_instruments == 3}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)++" "++(~instrument_3_full_midi.at(~piece_position).asString)};
			)
		)}, {"Bit pos. -- Midi --"});
		piecepositionstatictext.font = Font(size: ~firstviewboundsWIDTH/48);

		if(~pos_1 == nil, {~pos_1 = 0});
		if(~pos_2 == nil, {~pos_2 = 0});

		looptextfield_1 = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
		looptextfield_1.string = ~pos_1.asString;
		looptextfield_1.background_(~backgroundcolor);
		looptextfield_1.stringColor_(~stringcolor);
		looptextfield_1.font = Font(size: ~firstviewboundsWIDTH/48);
		looptextfield_1.action = {
			~sdfk = looptextfield_1.value.asInteger;
			/*	~pos_1 = looptextfield_1.value.asInteger;*/
			if(~sdfk.value.isInteger == true, {~pos_1 = ~sdfk.value.asInteger;}, {looptextfield_1.string = "integer".asString;});
			looptextfield_1.string = ~pos_1.asString;
		};

		looptextfield_2 = TextField(~firstview, Rect(~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
		looptextfield_2.string = ~pos_2.asString;
		looptextfield_2.background_(~backgroundcolor);
		looptextfield_2.stringColor_(~stringcolor);
		looptextfield_2.font = Font(size: ~firstviewboundsWIDTH/48);
		looptextfield_2.action = {
			~sdfk = looptextfield_2.value.asInteger;
			if(~sdfk.value.isInteger == true, {~pos_2 = ~sdfk.value.asInteger;}, {looptextfield_2.string = "integer".asString;});
			looptextfield_2.string = ~pos_2.asString;
		};

		/*~loop_creator_func = {

		if(("instrument_"++(1)++"_full_midi_before").asSymbol.envirGet == nil, {
		~number_of_instruments.do(t = 0; {
		t=t+1;
		("instrument_"++(t)++"_full_midi_before").asSymbol.envirPut(("instrument_"++(t)++"_full_midi").asSymbol.envirGet;);
		});
		});

		(
		h = 0.midicps;
		z = ~pos_1 - ~pos_2;
		if(z.isNegative == true, {z = z.neg;});
		z=z+1;
		~number_of_instruments.do(t = 0; {
		t=t+1;
		("loop_new_array"++(t)).asSymbol.envirPut([]);
		z.do(q = ~pos_1; {
		("loop_new_array"++(t)).asSymbol.envirPut(
		("loop_new_array"++(t)).asSymbol.envirGet.add(("instrument_"++(t)++"_full_midi").asSymbol.envirGet.at(q););

		);
		q=q+1;
		});

		("instrument_"++(t)++"_full_midi").asSymbol.envirPut(("loop_new_array"++(t)).asSymbol.envirGet;);

		});
		);

		~saved_piece_loaded = 1;
		~create_pseq_arrays_func.value;

		};

		/*~instrument_2_full_midi.size
		~instrument_2_full_midi.at(167)
		~instrument_2_full.at(167)
		~instrument_2_full.size*/

		loopsetbutton =  Button.new(~firstview, Rect(~firstviewboundsWIDTH/4.3, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21)).states_([["Set Loop",~stringcolor,~backgroundcolor]]).action_({arg button;
		~loop_on = 1;
		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~loop_creator_func.value;
		});

		stoploopbutton =  Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.9, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21)).states_([["Stop Loop",~stringcolor,~backgroundcolor]]).action_({arg button;
		~loop_on = 0;
		~number_of_instruments.do(t = 0; {
		t=t+1;
		("instrument_"++(t)++"_full_midi").asSymbol.envirPut(("instrument_"++(t)++"_full_midi_before").asSymbol.envirGet;);
		("instrument_"++(t)++"_full_midi_before").asSymbol.envirPut(nil);
		});

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~saved_piece_loaded = 1;
		~create_pseq_arrays_func.value;
		});*/

		if(~loop_on == nil, {~loop_on = 0});

		loopsetbutton =  Button.new(~firstview, Rect(~firstviewboundsWIDTH/4.3, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/8.5, ~firstviewboundsHEIGHT/21)).states_([["Loop is OFF",~stringcolor,~backgroundcolor],["Loop is ON",~stringcolor,~backgroundcolor]]).action_({arg button;

			if(button.value == 1, {~loop_on = 1;}, {~loop_on = 0;});
			/*if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
			~loop_creator_func.value;*/
		});

		if(~loop_on == 1, {loopsetbutton.value = 1}, {loopsetbutton.value = 0});

		if(~num_loops == nil, {~num_loops = 24});

		looptextfield_3 = TextField(~firstview, Rect(~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/1.47, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
		looptextfield_3.string = ~num_loops.asString;
		looptextfield_3.background_(~backgroundcolor);
		looptextfield_3.stringColor_(~stringcolor);
		looptextfield_3.font = Font(size: ~firstviewboundsWIDTH/48);
		looptextfield_3.action = {
			~sdfk = looptextfield_3.value.asInteger;
			if(~sdfk.value.isInteger == true, {~num_loops = ~sdfk.value.asInteger;}, {looptextfield_3.string = "integer".asString;});
			looptextfield_3.string = ~num_loops.asString;
		};

		/*ctrlperiodtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/13.5, ~firstviewboundsWIDTH/3.5, ~firstviewboundsHEIGHT/4.37));
		ctrlperiodtext.background=~backgroundcolor;
		ctrlperiodtext.string = "*Important Note*... You can kill all processes by hitting "++"cmd/ctrl - period".quote++"on your keyboard.";
		ctrlperiodtext.align = \topLeft;
		ctrlperiodtext.font = Font(size: ~firstviewboundsWIDTH/70.454545454545);*/

		newrandomseedsbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/5.1428571428571, ~firstviewboundsHEIGHT/21)).states_([["New Random Seeds",~stringcolor,~backgroundcolor],["New Random Seeds",~stringcolor,~backgroundcolor]]).action_({arg button;
			~randseed_1_backup = ~randseed_1;
			~randseed_2_backup = ~randseed_2;
			~randseed_3_backup = ~randseed_3;
			~randseed_1 =  99999999.rand;
			~randseed_2 =  99999999.rand;
			~randseed_3 =  99999999.rand;
			~save_random_seeds_func.value;
			z = "Current random seeds:\n"++
			"randseed 1 - "++~randseed_1++"\n"++
			"randseed 2 - "++~randseed_2++"\n"++
			"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton != 1, {
				"\n(also posted in post window)"++"\n"++
				"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
			z.postln;
			currentrandomseedstrext.string = z;
			currentrandomseedstrext.background=~backgroundcolor;
			currentrandomseedstrext.stringColor=~stringcolor;

		});
		newrandomseedsbutton.font = Font(size: ~firstviewboundsWIDTH/48);

		reversenewrandomseedsbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.1059907834101, ~firstviewboundsHEIGHT/6.2428571428571, ~firstviewboundsWIDTH/12, ~firstviewboundsHEIGHT/21)).states_([["Revert",~stringcolor,~backgroundcolor],["Revert",~stringcolor,~backgroundcolor]]).action_({arg button;
			~randseed_1_backup2 = ~randseed_1;
			~randseed_2_backup2 = ~randseed_2;
			~randseed_3_backup2 = ~randseed_3;
			~randseed_1 =  ~randseed_1_backup;
			~randseed_2 =  ~randseed_2_backup;
			~randseed_3 =  ~randseed_3_backup;
			~randseed_1_backup = ~randseed_1_backup2;
			~randseed_2_backup = ~randseed_2_backup2;
			~randseed_3_backup = ~randseed_3_backup2;
			~save_random_seeds_func.value;
			z = "Current random seeds:\n"++
			"randseed 1 - "++~randseed_1++"\n"++
			"randseed 2 - "++~randseed_2++"\n"++
			"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton != 1, {
				"\n(also posted in post window)"++"\n"++
				"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
			z.postln;
			currentrandomseedstrext.string = z;
			currentrandomseedstrext.background=~backgroundcolor;
			currentrandomseedstrext.stringColor=~stringcolor;
		});
		reversenewrandomseedsbutton.font = Font(size: ~firstviewboundsWIDTH/48);

		currentrandomseedstrext = TextView(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/4.6, ~firstviewboundsWIDTH/3.56, ~firstviewboundsHEIGHT/2.6));
		currentrandomseedstrext.background=~backgroundcolor;
		currentrandomseedstrext.string ="Current random seeds:\n"++
		"randseed 1 - "++~randseed_1++"\n"++
		"randseed 2 - "++~randseed_2++"\n"++
		"randseed 3 - "++~randseed_3.asString++(if(~quicknewpiecebutton != 1, {
			"\n(also posted in post window)"++"\n"++
			"*make sure to hit "++"generate piece".quote++" to generate a new piece based on the new seeds"}, {""}));
		/*currentrandomseedstrext.align = \topLeft;*/
		currentrandomseedstrext.stringColor=~stringcolor;
		currentrandomseedstrext.font = Font(size: ~firstviewboundsWIDTH/58.384615384615);

		see_saved_seeds_file_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.6, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21)).states_([["See Saved Random Seeds",~stringcolor,~backgroundcolor],["See Saved Random Seeds",~stringcolor,~backgroundcolor]]).action_({arg button;

			h = Document.open(Document.current.dir++"/duranduranbot_saved_random_seeds.scd");


		});
		see_saved_seeds_file_button.font = Font(size: ~firstviewboundsWIDTH/48);

		newpieceautobutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.25, ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21)).states_([["Auto New Piece On",~stringcolor,~backgroundcolor],["Auto New Piece Off",~stringcolor,~backgroundcolor]]).action_({arg button;
			if(button.value == 1, {
				g = File(Document.current.dir++"/other_program_files/autonewpiece.scd", "w");
				g.write("0");
				g.close;
				newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is OFF";
			}, {
				newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is ON";
				g = File(Document.current.dir++"/other_program_files/autonewpiece.scd", "w");
				g.write("1");
				g.close;
			});

		});
		newpieceautobutton.font = Font(size: ~firstviewboundsWIDTH/48);

		newpieceautotext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/4.5, ~firstviewboundsHEIGHT/1.25, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
		newpieceautotext.background=~backgroundcolor;
		newpieceautotext.stringColor=~stringcolor;
		if((FileReader.read(Document.current.dir++"/other_program_files/autonewpiece.scd", false, true).flatten.at(0) == 1.asString), {
			newpieceautobutton.value = 0; newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is ON";
		}, {newpieceautobutton.value = 1; newpieceautotext.string = "Automatic "++"Quick New Piece".quote++" is OFF";});
		newpieceautotext.font = Font(size: ~firstviewboundsWIDTH/55);

		machinelearningbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.08, ~firstviewboundsWIDTH/5.5, ~firstviewboundsHEIGHT/21)).states_([["ML Training Is ON",~stringcolor,~backgroundcolor],["ML Training Is OFF",~stringcolor,~backgroundcolor]]).action_({arg button;
			if(button.value == 1, {
				g = File(Document.current.dir++"/other_program_files/ddbot_machine_learning_on_off.scd", "w");
				g.write("0");
				g.close;
			}, {
				g = File(Document.current.dir++"/other_program_files/ddbot_machine_learning_on_off.scd", "w");
				g.write("1");
				g.close;
			});

		});
		machinelearningbutton.font = Font(size: ~firstviewboundsWIDTH/48);

		machinelearningtext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/1.08, ~firstviewboundsWIDTH/2, ~firstviewboundsHEIGHT/21));
		machinelearningtext.background=~backgroundcolor;
		machinelearningtext.stringColor=~stringcolor;
		machinelearningtext.string = "More info about machine learning on "++"instructions".quote++" page";
		machinelearningtext.font = Font(size: ~firstviewboundsWIDTH/55);

		q = FileReader.read(Document.current.dir++"/other_program_files/ddbot_machine_learning_on_off.scd");
		2.do({q = q.at(0)});
		q = q.asInteger;
		if(q == 1, {machinelearningbutton.value = 0;});

		loadscorebuttonbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.16, ~firstviewboundsWIDTH/5.8, ~firstviewboundsHEIGHT/21)).states_([["Load Saved Score",~stringcolor,~backgroundcolor]]).action_({arg button;

			if(~loaded_score.value == nil, {
				loadscoretext.string = "Paste score in top of "++"duranduranbot_RUN.scd".quote++" file in "++"~loaded_score".quote++" function.";
				AppClock.sched(5, {loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";});
			}, {
				~saved_piece_loaded = 1;
				~loaded_score.value;
				if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
				generatepiecebutton.valueAction = 1;
				loadscoretext.string = "Score has been loaded. Press "++"Play Piece".quote++" to run.";
				AppClock.sched(5, {loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";});
			});

		});
		loadscorebuttonbutton.font = Font(size: ~firstviewboundsWIDTH/48);

		loadscoretext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/5.2, ~firstviewboundsHEIGHT/1.16, ~firstviewboundsWIDTH/1.3, ~firstviewboundsHEIGHT/21));
		loadscoretext.background=~backgroundcolor;
		loadscoretext.stringColor=~stringcolor;
		loadscoretext.string = "Load a saved score here. Then press "++"Play Piece".quote++".";
		loadscoretext.font = Font(size: ~firstviewboundsWIDTH/55);

		training_settings_page_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.45, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21)).states_([["Player Settings",~stringcolor,~backgroundcolor]]).action_({arg button;
			~player_settings_page_func.value;
		});
		training_settings_page_button.font = Font(size: ~firstviewboundsWIDTH/48);

		playerchoicetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/12, ~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/21));
		playerchoicetext.background=~backgroundcolor;
		playerchoicetext.stringColor=~stringcolor;
		playerchoicetext.string = "Current Player(s) - "++(
			case
			{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 0.asString}{"All Players"}
			{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 1.asString}{"Player 1"}
			{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 2.asString}{"Player 2"}
			{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 3.asString}{"Player 3"};
		)++", "++(
			case
			{FileReader.read(Document.current.dir++"/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 2.asString}{"2 voices"}
			{FileReader.read(Document.current.dir++"/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 3.asString}{"3 voices"};
		);
		playerchoicetext.align = \topLeft;
		playerchoicetext.font = Font(size: ~firstviewboundsWIDTH/55);

		~color_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.33, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
		~color_menu.items = [
			"GUI color - "++(
				case
				{FileReader.read(Document.current.dir++"/other_program_files/gui_color.scd", false, true).flatten.at(0) == 0.asString}{"Black"}
				{FileReader.read(Document.current.dir++"/other_program_files/gui_color.scd", false, true).flatten.at(0) == 1.asString}{"White"};
			),
			"Black", "White"];
		~color_menu.background_(~backgroundcolor);
		~color_menu.stringColor_(~stringcolor);
		~color_menu.value = FileReader.read(Document.current.dir++"/other_program_files/gui_color.scd", false, true);
		~color_menu.action = { arg menu;
			g = File(Document.current.dir++"/other_program_files/gui_color.scd", "w");
			g.write(
				case
				{~color_menu.value == 0}{nil}
				{~color_menu.value == 1}{"0".asString}
				{~color_menu.value == 2}{"1".asString};

			);
			g.close;
			case
			{~color_menu.value == 0}{nil}
			{~color_menu.value == 1}{
				~backgroundcolor = Color.black;
				~stringcolor = Color.white;
			}
			{~color_menu.value == 2}{
				~backgroundcolor = Color.white;
				~stringcolor = Color.black;
			};
			w.close;
			~duranduranbot_GUI_func.value;
			~main_page_func.value;
		};
		~color_menu.font = Font(size: ~firstviewboundsWIDTH/48);

		save_current_piece_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.4173228346457, ~firstviewboundsHEIGHT/1.23, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21)).states_([["Save Current Piece",~stringcolor,~backgroundcolor]]).action_({arg button;

			"please wait".postln;
			g = File(Document.current.dir++"/other_program_files/ddbot_saved_scores/score_"++(z=((Date.getDate.format("%d_%m_%Y__%H_%M_%S").asString)))++".scd", "w");
			g.write("// Copy and paste the all lines below in the "++"Load Saved Score".quote++" section in the top of the "++"duranduranbot_RUN.scd".quote++" file");
			g.close;
			~re = Routine({
				~number_of_instruments.do(x = 0; {
					g = File(Document.current.dir++"/other_program_files/ddbot_saved_scores/score_"++(z)++".scd", "a");
					g.write("\n~instrument_"++(x=x+1)++"_full_midi = [ ");
					g.close;
					0.25.wait;
					y = (-1);
					g = File(Document.current.dir++"/other_program_files/ddbot_saved_scores/score_"++(z)++".scd", "a");
					(("instrument_"++(x)++"_full_midi").asSymbol.envirGet.size-1).do({
						g.write((("instrument_"++(x)++"_full_midi").asSymbol.envirGet.at(y=y+1)).asString++", ");

					});
					g.close;
					0.25.wait;
					g = File(Document.current.dir++"/other_program_files/ddbot_saved_scores/score_"++(z)++".scd", "a");
					g.write((("instrument_"++(x)++"_full_midi").asSymbol.envirGet.at(y=y+1)).asString++" ];");
					g.close;
				});
				"score saved".postln;
				("see scores in the "++"program files".quote++" folder").postln;
				~re.stop;
			});
			~re.play;

		});
		save_current_piece_button.font = Font(size: ~firstviewboundsWIDTH/48);

		closeddbotbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/1.1, ~firstviewboundsWIDTH/7.5, ~firstviewboundsHEIGHT/21)).states_([["Close DD'bot",~stringcolor,~backgroundcolor],["Close DD'bot",~stringcolor,~backgroundcolor]]).action_({arg button;
			~stop_func.value;
			Server.freeAll;
			w.close;
		});
		closeddbotbutton.font = Font(size: ~firstviewboundsWIDTH/48);

	};

	~instructions_page_func = {

		~firstview.close;

		~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
		/*view.clearOnRefresh = false;*/
		~firstview.background = ~backgroundcolor;
		~firstviewboundsWIDTH = ~firstview.bounds.width;
		~firstviewboundsHEIGHT = ~firstview.bounds.height;

		firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21));
		firstviewtext1.background=~backgroundcolor;
		firstviewtext1.stringColor=~stringcolor;
		firstviewtext1.string = "Instructions";
		firstviewtext1.align = \topLeft;
		firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
			~firstview.close;
			~main_page_func.value;
		});
		firstviewtext2.background=~stringcolor;
		firstviewtext2.string = "Back";
		firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/1.5652173913043, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/2.88, ~firstviewboundsHEIGHT/4.37));
		firstviewtext3.background=~backgroundcolor;
		firstviewtext3.stringColor=~stringcolor;
		firstviewtext3.string = "*Important Note*... You can kill all sounds and processes by hitting "++"cmd/ctrl - period".quote++"on your keyboard.";
		firstviewtext3.align = \topLeft;
		firstviewtext3.font = Font(size: ~firstviewboundsWIDTH/65.454545454545);

		firstviewtext4 = TextView(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/7.528571428571, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/1.2));
		firstviewtext4.background=~backgroundcolor;
		firstviewtext4.string = "(Scroll down for more)\n\nHOW TO PLAY A PIECE\n\nPlay a DuranDuranbot piece by clicking the "++"generate piece".quote++"  button on the Main Page, and then by clicking the "++"play piece".quote++"  button. Once random seeds are input into the program on the front page (the very first page of this program), you only need to generate a song once using the same random seeds. If you click on the "++"generate piece".quote++"  button more than once without inputting new random seeds the same piece will always be generated. To generate a new piece, you can go back to the first page and input new random seeds, or you can do this automatically by clicking on the "++"new random seeds".quote++" button across from the "++"generate piece".quote++" button on the right side of the main page. Or, you can quickly generate a new piece based on entirely new random seeds by simply clicking on the "++"quick new piece".quote++" button a few sets of buttons below the "++"generate piece".quote++" button. \n\nMACHINE LEARNING\n\nMachine learning is enabled by default, but you can disable it on the main page. Using the machine learning function allows for you to teach DuranDuranbot what you prefer to hear in a piece by rating a piece, and for it become better at composing music for you. On the Main Page, you can rate a piece by clicking on any of the point ratings and clicking "++"Rate".quote++". The rating system is currently set to rate the last "++(~auto_select_diff)++" seconds from your current position in the piece designated by where the time slider is positioned... Doing all of this will train DuranDuranbot how to compose music. All training data will be saved to the DuranDuranbot "++"ddbot_training_data".quote++" folder for future use. (Side Note: You can hybridize the training data of numerous DD'bots as well by going into the DD'bot training folder.)\n\nDDbot's machine learning system utilizes mathematical weights to compose/learn to compose music. For instance, a piece compostition variable might be a note interval, and each note interval (1 step, 2 steps) would have a weight assigned to it in a given situation... and DDbot would choose an interval by comparing the weight values of each interval in a given situation, *tending* to choose a higher weighted variable using weighted probability. When you rate a section of a piece while using DDbot, a variable in a given situation is rewarded with points as to increase the value of it's weight. The more points you give a variable, the more chance it will be used in a piece composition. How powerful each xpt rating is - x16 pts is 80% weighted chance for variable in 5 reward instances for that variable... x8 pts is 80% weighted chance for variable in 10 reward instances. x4 pts is 80% weighted chance for variable in 20 reward instances.... and so on in that same pattern. Put simply, a higher rating point value equals faster training of DDbot, but less fine-tuning.\n\nPLAYER SETTINGS\n\nBy clicking the "++"Player Settings".quote++" button on the Main Page, you can choose which player(s) you can traget to compose music and be trained.\n\nLOOPING\n\nCurrently unmarked - 1st text box is starting bit position, 2nd text box is ending bit position. Last text box is how many loops.\n\nCURRENT BUGS\n\n- Every time you click on a link to the main page while piece is playing, the piece goes to it's beginning for a slight moment then goes back to current position.\n\n\n\nIf you have any questions, please feel free to contact -\n\nken_brant@ymail.com\n\nor\n\nkenbrant7@gmail.com";
		firstviewtext4.stringColor=~stringcolor;
		firstviewtext4.font = Font(size: ~firstviewboundsWIDTH/48);

	};

	~player_settings_page_func = {

		~firstview.close;

		~firstview = View(w, Rect(0, 0, w.bounds.width, w.bounds.height));
		/*view.clearOnRefresh = false;*/
		~firstview.background = ~backgroundcolor;
		~firstviewboundsWIDTH = ~firstview.bounds.width;
		~firstviewboundsHEIGHT = ~firstview.bounds.height;

		firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21));
		firstviewtext1.background=~backgroundcolor;
		firstviewtext1.stringColor=~stringcolor;
		firstviewtext1.string = "Player Settings";
		firstviewtext1.align = \topLeft;
		firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
			~firstview.close;
			~main_page_func.value;
		});
		firstviewtext2.string = "Back";
		firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

		playerspeedtextfield = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
		playerspeedtextfield.string = ~bit_duration.reciprocal.asString;
		playerspeedtextfield.background_(~backgroundcolor);
		playerspeedtextfield.stringColor_(~stringcolor);
		playerspeedtextfield.font = Font(size: ~firstviewboundsWIDTH/48);
		playerspeedtextfield.action = {
			/*randseed_1_make = randseedEZText1.value;
			randseed_1_make = randseed_1_make.asString++"";
			randseed_1_make = randseed_1_make.separate;*/
			/*if(i.size > 8, {num = nil; playerspeedtextfield.string = "must be 8 digits or less";}, {*/
			~bit_duration = playerspeedtextfield.value.asFloat.reciprocal;
			g = File(Document.current.dir++"/other_program_files/bit_speed.scd", "w");
			g.write(~bit_duration.asString);
			g.close;
			playerspeedtextfield2.string = "Current - "++(~bit_duration.reciprocal)++" bits per second ... Input how fast "++"bits".quote++" are cycled through. Then press "++"Enter/Return".quote++". Original default is "++(~bit_duration_default)++".";

			/*AppClock.sched(0, {
			~fullpiecedur1 = ((~piecedur_secs = (("instrument_"++(1)++"_full").asSymbol.envirGet.size*~bit_duration).round)/60);
			~dd_secs = ((~fullpiecedur1-(~dd_mins = ~fullpiecedur1.roundUp-1))*60).roundUp;
			/*~fullpiecedur = ("piece length - "++(~dd_mins.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asString)++" mins");
			*/
			~fullpiecedur = ("Piece Length - "++(~piecedur_secs.asInteger.asString)++" secs ("++(~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString)++" mins)");
			piecedurationstatictext.string = ~fullpiecedur;
			("\npiece duration: \n"++((~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString))++" minutes";).postln;
			/*rangeslider.set(spec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"), initVal: [0,if(~piecedur_secs != nil, {~piecedur_secs}, {0})]);
			*/timeslider.set(spec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"));
			~rate_selection_start = 0;
			~rate_selection_end = ("instrument_"++(1)++"_full").asSymbol.envirGet.size;
			});*/
			/*randseedviewtext1.string = if(randseed_1_make.isNumber == true, {
			randseed_1_make = randseed_1_make.asFloat; randseed_1_make.postln; ~randseed_1 = randseed_1_make; ~save_random_seeds_func.value; ~randseed_1.asString++" entered";}, {
			randseed_1_make = nil; randseedEZText1.string = "Enter Numerical Value";
			});
			});*/
		};

		playerspeedtextfield2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/1.3, ~firstviewboundsHEIGHT/11));
		playerspeedtextfield2.background=~backgroundcolor;
		playerspeedtextfield2.stringColor=~stringcolor;
		playerspeedtextfield2.string = "Current - "++(~bit_duration.reciprocal)++" bits per second ... Input how fast "++"bits".quote++" are cycled through. Then press "++"Enter/Return".quote++". Original default is "++(~bit_duration_default)++".";
		playerspeedtextfield2.font = Font(size: ~firstviewboundsWIDTH/48);

		~num_instruments_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/3.75, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
		~num_instruments_menu.items = [
			"Current - "++(
				case
				{FileReader.read(Document.current.dir++"/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 2.asString}{"2 voices"}
				{FileReader.read(Document.current.dir++"/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 3.asString}{"3 voices"};
			),
			"2 voices", "3 voices"];
		~num_instruments_menu.background_(~backgroundcolor);
		~num_instruments_menu.stringColor_(~stringcolor);
		~num_instruments_menu.value = FileReader.read(Document.current.dir++"/other_program_files/numberofvoices.scd", false, true);
		~num_instruments_menu.action = { arg menu;

			~stop_func.value;
			w.close;
			Server.freeAll;
			~duranduranbot_routine.stop;

			g = File(Document.current.dir++"/other_program_files/numberofvoices.scd", "w");
			case
			{~num_instruments_menu.value == 0}{nil}
			{~num_instruments_menu.value == 1}{g.write("2".asString); g.close; ~number_of_instruments = 2;}
			{~num_instruments_menu.value == 2}{g.write("3".asString); g.close; ~number_of_instruments = 3;};

			~duranduranbot_GUI_func.value;
		};
		~num_instruments_menu.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.6, ~firstviewboundsHEIGHT/3.75, ~firstviewboundsWIDTH/1.4, ~firstviewboundsHEIGHT/9));
		firstviewtext3.background=~backgroundcolor;
		firstviewtext3.stringColor=~stringcolor;
		firstviewtext3.string = "Set the default setting of how many voices/intruments are playing. Original default is 2.";
		firstviewtext3.align = \topLeft;
		firstviewtext3.font = Font(size: ~firstviewboundsWIDTH/48);

		~players_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/1.75, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
		~players_menu.items = [
			"Current - "++(
				case
				{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 0.asString}{"All Players"}
				{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 1.asString}{"Player 1"}
				{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 2.asString}{"Player 2"}
				{FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 3.asString}{"Player 3"};
			),
			"All Players", "Player 1", "Player 2", "Player 3"];
		~players_menu.background_(~backgroundcolor);
		~players_menu.stringColor_(~stringcolor);
		~players_menu.value = FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true);
		~players_menu.action = { arg menu;
			g = File(Document.current.dir++"/other_program_files/which_players.scd", "w");
			g.write(
				case
				{~players_menu.value == 0}{nil}
				{~players_menu.value == 1}{"0".asString}
				{~players_menu.value == 2}{"1".asString}
				{~players_menu.value == 3}{"2".asString}
				{~players_menu.value == 4}{"3".asString};

			);
			g.close;
		};
		~players_menu.font = Font(size: ~firstviewboundsWIDTH/48);

		firstviewtext4 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.5, ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/6.5));
		firstviewtext4.background=~backgroundcolor;
		firstviewtext4.stringColor=~stringcolor;
		firstviewtext4.string = "Click on the pop-up menu below to choose which player(s) will compose and be trained. This is useful for a more targeted composition and training... or less targeted. By selecting an option from the pop-up menu below, you are making that the default setting. Original default is "++"Player 1".quote++".";
		firstviewtext4.align = \topLeft;
		firstviewtext4.font = Font(size: ~firstviewboundsWIDTH/48);
	};

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// HOTKEYS

	~spacebar_unicode = 32; //pause DDbot

	w.view.keyDownAction = { arg view, char, modifiers, unicode, keycode;
		[char, modifiers, unicode, keycode];

		//pause/unpause synth
		if(unicode == ~spacebar_unicode, {
			if(~duranduranbot_routine.isPlaying == true, {
				if(~pause_piece == 0, {~playbutton.valueAction = 0});
				if(~pause_piece == 1, {~playbutton.valueAction = 1});
			});
		});
	};

	//To find key codes in post window... Hightlight the function and its parenthesis (without the /* and */) in red below,
	//and press "ctrl - enter" or "cmd - return" - then a small window called "test" will pop up in the taskbar...
	//Press the "type" button inside of the window, then press anywhere on the keyboard to find
	//the key codes in the right-hand side post window. Layout: [unicode, keycode, modifiers]. It is either a unicode
	//or a keycode. Do not permanently delete the /* and */ characters, or else the
	//program will not run.

	/*

	(
	var win = Window(\test, Rect(10, 100, 100, 50)),
	btn = Button(win, Rect(5, 5, 90, 40)).states_([["type"]]);
	btn.keyDownAction =
	{ arg view, char, modifiers, unicode, keycode;
	["unicode:"+unicode, "keycode:"+keycode, "modifiers:"+modifiers].postln;
	};
	win.front;
	)

	*/

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// SAVE RANDOM SEEDS TO FILE

	~save_random_seeds_func = {

		a = Date.getDate;

		/*Document.open(Document.current.dir++"/duranduranbot_saved_random_seeds.scd").close;*/

		h = File.new(Document.current.dir++"/duranduranbot_saved_random_seeds.scd", "a+");

		h.write("\n\n"++a.asString++
			"\nrandom seed 1 - "++~randseed_1.asString++
			"\nrandom seed 2 - "++~randseed_2.asString++
			"\nrandom seed 3 - "++~randseed_3.asString
		);

		h.close;
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// BUILD/GENERATE/COMPOSE PIECE

	~clock_diff_0_func = {
		~clock_diff_0 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet - ("old_instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet;
		if(~clock_diff_0.isFloat == true, {~clock_diff_0 = ~clock_diff_0.round});
		if(~clock_diff_0.isNegative == true, {~clock_diff_0 = ~clock_diff_0.neg});
		while({~clock_diff_0 > 12}, {~clock_diff_0 = ~clock_diff_0 - 12});
	};

	~clock_diff_1_func = {
		~clock_diff_1 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet - ("instrument_"++(~inst_array.wrapAt(~inst_num))++"_freq").asSymbol.envirGet;
		if(~clock_diff_1.isFloat == true, {~clock_diff_1 = ~clock_diff_1.round});
		if(~clock_diff_1.isNegative == true, {~clock_diff_1 = ~clock_diff_1.neg});
		while({~clock_diff_1 > 12}, {~clock_diff_1 = ~clock_diff_1 - 12});
	};

	~clock_diff_2_func = {
		~clock_diff_2 = ("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirGet - ("instrument_"++(~inst_array.wrapAt(~inst_num+1))++"_freq").asSymbol.envirGet;
		if(~clock_diff_2.isFloat == true, {~clock_diff_2 = ~clock_diff_2.round});
		if(~clock_diff_2.isNegative == true, {~clock_diff_2 = ~clock_diff_2.neg});
		while({~clock_diff_2 > 12}, {~clock_diff_2 = ~clock_diff_2 - 12});
	};

	~note_generator_func = {

		case
		{~number_of_instruments == 2}{2.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});}
		{~number_of_instruments == 3}{3.do(s = (-1); {("clock_diff_"++(s=s+1)++"_func").asSymbol.envirGet.value;});};

		(~number_of_instruments-1).do(u = (-1); {
			u=u+1;
			("clock_diff_"++(u)).asSymbol.envirPut(("clock_diff_"++(u)).asSymbol.envirGet.round.asInteger;);
		});

		~intv_weights = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(~player_array.wrapAt(~inst_num-1))++"_interval_weights/"++(
			case
			{~number_of_instruments == 2}{(~clock_diff_0.asString)++"__"++(~clock_diff_1.asString)}
			{~number_of_instruments == 3}{(~clock_diff_0.asString)++"__"++(~clock_diff_1.asString)++"_"++(~clock_diff_2.asString)};
		)++"_weights.scd", false, true);
		~intv_weights = ~intv_weights.flatten.asFloat;
		~intv_weights = ~intv_weights.normalizeSum;
		~new_interval = ~interval_array.wchoose(~intv_weights);
		~new_interval = [~new_interval, ~new_interval.neg].choose;

		("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet + ~new_interval);

		if(((("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet > 90) or: (("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet < 55)), {
			while({("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet > 90}, {("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet-12)});
			while({("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet < 55}, {("instrument_"++(~inst_num)++"_freq").asSymbol.envirPut(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet+12)});
		});

		("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;
	};

	~time_generator_func = {

		(~number_of_instruments).do(e = 0; {
			e=e+1;
			case
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(0)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(1))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(0)+~time_array.at(1)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(0)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(1)})}
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(1)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(2))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(1)+~time_array.at(2)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(1)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(2)})}
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(2)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(3))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(2)+~time_array.at(3)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(2)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(3)})}
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(3)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(4))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(3)+~time_array.at(4)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(3)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(4)})}
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(4)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(5))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(4)+~time_array.at(5)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(4)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(5)})}
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(5)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(6))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(5)+~time_array.at(6)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(5)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(6)})}
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(6)) and: (("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration <= ~time_array.at(7))}{if((("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration < (~time_array.at(6)+~time_array.at(7)/2)), {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(6)}, {("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(7)})}
			{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_choice").asSymbol.envirGet*~bit_duration >= ~time_array.at(7))}{("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+e))++"_place").asSymbol.envirPut(7)};
		});


		/*
		("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+2))++"_choice").asSymbol.envirGet
		("dna_clock_"++(~inst_array.wrapAt(~inst_num-1+2))++"_place").asSymbol.envirGet
		*/

		~time_weights = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(~player_array.wrapAt(~inst_num-1))++"_time_weights/"++(
			case
			{~number_of_instruments == 2}{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet.asString)++"_"++(("dna_clock_"++(~inst_array.wrapAt(~inst_num))++"_place").asSymbol.envirGet.asString)}
			{~number_of_instruments == 3}{(("dna_clock_"++(~inst_array.wrapAt(~inst_num-1))++"_place").asSymbol.envirGet.asString)++"_"++(("dna_clock_"++(~inst_array.wrapAt(~inst_num))++"_place").asSymbol.envirGet.asString)++"_"++(("dna_clock_"++(~inst_array.wrapAt(~inst_num+1))++"_place").asSymbol.envirGet.asString)};
		)++"_time_weights.scd", false, true);
		~time_weights = ~time_weights.flatten.asFloat;
		~time_weights = ~time_weights.normalizeSum;
		~new_time = ~dna_choice_array.wchoose(~time_weights);

		("dna_clock_"++(~inst_num)++"_choice").asSymbol.envirPut(~new_time);
	};

	~inst_clock_func = {

		~number_of_instruments.do(n = 0; {("dna_clock_"++(n=n+1)).asSymbol.envirPut(("dna_clock_"++(n)).asSymbol.envirGet+1);});
		~number_of_instruments.do(n = 0; {
			n=n+1;
			~inst_num = n;
			("instrument_"++(~inst_num)++"_array").asSymbol.envirPut(
				("instrument_"++(~inst_num)++"_array").asSymbol.envirGet.add(
					if(("dna_clock_"++(~inst_num)).asSymbol.envirGet == ("dna_clock_"++(~inst_num)++"_choice").asSymbol.envirGet, {
						("dna_clock_"++(~inst_num)).asSymbol.envirPut((-1));
						("instrument_"++(~inst_num)++"_if_zero").asSymbol.envirPut("false");
						~note_generator_func.value; //find new interval for current instrument
						~time_generator_func.value; //find new bit duration for current instrument
						~put1 = ("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;
					}, {
						~put1 = if(("instrument_"++(~inst_num)++"_if_zero").asSymbol.envirGet == "false", {("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;}, {0});
					});
					~put1.value;
				);
			);
		});
	};


	~piece_generator_func = {

		// THERE IS A POINT TO USING BIT-BY-BIT CIRCULAR COMPOSITION ( https://scsynth.org/t/bit-by-bit-circular-composition/1107 )
		// Not using it, DDbot player can't compose based on everyone else, for the player isn't composing in parallel.
		// Players NEED to be composing in parallel with everyone else... On the "same track".

		(
			(

				"gen checkpoint 1".post;

				~tuning_frequency = 440; //default 440 hz
				~freqmap = (i = ~tuning_frequency / (2 ** (69 / 12)); a = (-1); b = (-1); Array.fill(128, {i * (2 ** (((a=a+1) / 12).trunc)) * ( (2 ** (12.reciprocal)) ** ((b=b+1)  % 12));}));/*~freqmap = (i = 8.0271480262684/*-0.0000000000024*/; ~stepratio = 1.0594630943593; Array.fill(128, { i = i*~stepratio;}));*/
				~freqmap = ~freqmap.cpsmidi;
				/*~freqmap2 = (i = ~freqmap.at(58)/*-0.0000000000024*/; ~stepratio = 1.0594630943593; Array.fill(24, { i = i*~stepratio;}));*/
				~freqmap2 = (n = 54; Array.fill(36, {~freqmap.at(n=n+1);}));
				/*~ratio_array = (x = 0; Array.fill(12, {1.0594630943593**(x=x+1)}));*/
				~interval_array = (1..12);
				~midi_ratio_array = (1..12);
				~midi_high_limit = 90;
				~midi_low_limit = 55;

				thisThread.randSeed = ~randseed_1;

				~player_array = (x = 0; Array.fill(if(FileReader.read(Document.current.dir++"/other_program_files/which_players.scd", false, true).flatten.at(0) == 0.asString, {~number_of_instruments}, {1}), {"player_"++(x=x+1)}));
				~player_array = ~player_array.scramble;
				~number_of_instruments.do(x = 0; {("inst_"++(x=x+1)).asSymbol.envirPut(~player_array.at(x-1));});

				/*~dna_choice_array = (1..64);*/
				~time_array = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8];
				~dna_choice_array = [1, 2, 4, 8, 16, 32, 64, 128];
				~dna_choice_array_num = (1..~dna_choice_array.size);

				x = 0;
				~inst_array = Array.fill(~number_of_instruments, {x=x+1});



				if(~duranduranbot_routine.isPlaying == true, {~number_of_instruments.do(x = 0; {("duranduranbotsynth"++(x=x+1)).asSymbol.envirGet.set(\freq, 0);});});


				~number_of_instruments.do(x = 0; {
					x=x+1;
					~number_of_parts_in_piece.do(y = 0; {("instrument_"++(x)++"_array_part_"++(y=y+1)).asSymbol.envirPut([]);});
				});

			);

			" 2".post;

			//Piece creation

			(

				~create_piece_function = {

					~number_of_instruments.do(x = 0; {
						x=x+1;
						("instrument_"++(x)++"_full").asSymbol.envirPut([]);
						("instrument_"++(x)++"_array").asSymbol.envirPut([]);
					});

					// choose compoition process

					if(~composition_process_setting == nil, {~composition_process_setting = 1});

					case
					{~composition_process_setting == 1}{

						// I NEED DDBOT TO SOUND *KINDA ROBOTIC, AND KINDA HUMAN*


						// Top level -
						// Generate one random chord. Having nothing to do with time, generate follow up chords in the sequence using ML data.
						// Level 3 -
						// For each chord in above level sequence, do same thing as top level.
						// Level 2 -
						// " "
						// Level 1 (when ~create_piece_function fires) -
						// For each chord

						// For now... override

						// Have DDbot machine learn preferences in that^


						// generate top-level chords

						~number_of_instruments.do(x = 0; {
							x=x+1;
							("top_level_notes_"++(x)).asSymbol.envirPut([]);
						});

						thisThread.randSeed = ~randseed_2;

						~number_of_instruments.do(n = 0; {
							~inst_num = (n=n+1);
							("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirPut([rrand(~midi_low_limit, ~midi_high_limit)]);
							("instrument_"++(~inst_array.wrapAt(~inst_num-1))++"_freq").asSymbol.envirPut(("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirGet.at(0).asInteger);
						});

						~number_of_instruments.do(s = 0; {("old_instrument_"++(s=s+1)++"_freq").asSymbol.envirPut(("top_level_notes_"++(s)).asSymbol.envirGet.at(0).asInteger + 12)});

						/*n = 3; x = Array.fill(13, {n=n+1});
						y = Array.fill(x.size, {1/x.size}).normalizeSum;
						z = (x.wchoose(y.value)-1);
						(if(z.odd == true, {[z=z+[1,(-1)].choose, z].wchoose([2/3,1/3].normalizeSum)}))*/
						(exprand(4, 16).round-1).do({
							~number_of_instruments.do(n = 0; {
								~inst_num = (n=n+1);
								~note_generator_func.value;
								("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirPut(("top_level_notes_"++(~inst_array.wrapAt(~inst_num-1))).asSymbol.envirGet.add(("instrument_"++(~inst_num)++"_freq").asSymbol.envirGet;););
							});
						});

						y = rrand(2, 8);
						~number_of_instruments.do(x = 0; {
							("top_level_notes_"++(x=x+1)++"_full").asSymbol.envirPut([]);
							y.do({
								("top_level_notes_"++(x)++"_full").asSymbol.envirPut(("top_level_notes_"++(x)++"_full").asSymbol.envirGet.add(("top_level_notes_"++(x)).asSymbol.envirGet;));
							});
							("top_level_notes_"++(x)++"_full").asSymbol.envirPut(("top_level_notes_"++(x)++"_full").asSymbol.envirGet.flatten);
						});



						/*
						("top_level_notes_"++(3)++"_full").asSymbol.envirGet
						~top_level_notes_1.postln;
						~top_level_notes_2.postln;
						~top_level_notes_3.postln;

						("instrument_"++(1)++"_freq").asSymbol.envirGet
						*/


						// generate bottom-level chords/piece

						thisThread.randSeed = ~randseed_3;

						~number_of_instruments.do(n = 0; {("dna_clock_"++(n=n+1)).asSymbol.envirPut((-1));});
						~number_of_instruments.do(n = 0; {("dna_clock_"++(n=n+1)++"_choice").asSymbol.envirPut(~dna_choice_array.choose)});
						~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});
						/*~number_of_instruments.do(n = 0; {("bottom_level_notes_"++(n=n+1)).asSymbol.envirPut([])});*/

						~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_full").asSymbol.envirPut([]);});

						~first_comp_part_func = {

							// "put" top level chords into ~instrument_n_freq's

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_freq").asSymbol.envirPut(("top_level_notes_"++(n)++"_full").asSymbol.envirGet.at(x).asInteger);
							});

							// run composition process with times/durations

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_array").asSymbol.envirPut([]);
								("instrument_"++(n)++"_array_1").asSymbol.envirPut([]);
							});
							(o = (exprand(4, 16).round)/~bit_duration).do({~inst_clock_func.value});
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_1").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

						};

						~composition_setting = 2;


						//compose action

						(~top_level_notes_1_full.size-1).do(x = (-1); ~part_num = 0; {
							x=x+1;
							~part_num = ~part_num + 1;


							if(~composition_setting == 1, {~first_comp_part_func.value}, {
								if(x > 0, {

									~number_of_instruments.do(n = 0; {
										n=n+1;
										("instrument_"++(n)++"_array_1").asSymbol.envirPut(
											case
											{~composition_setting == 2}{("instrument_"++(n)++"_array_2_forwards").asSymbol.envirGet;}
											{~composition_setting == 3}{("instrument_"++(n)++"_array_blended").asSymbol.envirGet;};
										);
									});
								}, {~first_comp_part_func.value});
							});


							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_freq").asSymbol.envirPut(("top_level_notes_"++(n)++"_full").asSymbol.envirGet.at(x+1).asInteger);
							});

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_array").asSymbol.envirPut([]);
								("instrument_"++(n)++"_array_2").asSymbol.envirPut([]);
							});
							o.do({~inst_clock_func.value});
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet)});

							// reverse array 2, then blend
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2_forwards").asSymbol.envirPut(("instrument_"++(n)++"_array_2").asSymbol.envirGet)});
							~number_of_instruments.do(n = 0; {("instrument_"++(n=n+1)++"_array_2").asSymbol.envirPut(("instrument_"++(n)++"_array_2").asSymbol.envirGet.reverse)});

							~number_of_instruments.do(n = 0; {n=n+1; ("instrument_"++(n)++"_array_blended").asSymbol.envirPut([]);});

							~number_of_instruments.do(n = 0; {
								n=n+1;
								o.do(i = (-1); {
									i=i+1;
									u = ((("instrument_"++(n)++"_array_1").asSymbol.envirGet.at(i) + ("instrument_"++(n)++"_array_2").asSymbol.envirGet.at(i)) / 2).round;
									("instrument_"++(n)++"_array_blended").asSymbol.envirPut(("instrument_"++(n)++"_array_blended").asSymbol.envirGet.add(u.value));
								});
							});

							~number_of_instruments.do(n = 0; {
								n=n+1;
								("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_full").asSymbol.envirGet.add(("instrument_"++(n)++"_array_blended").asSymbol.envirGet;));
								("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_full").asSymbol.envirGet.flatten);
							});

						});


						/*	("instrument_"++(2)++"_full").asSymbol.envirGet;*/


						/*	("instrument_"++(1)++"_array_2").asSymbol.envirGet;
						("instrument_"++(2)++"_array_2").asSymbol.envirGet;
						("instrument_"++(3)++"_array_2").asSymbol.envirGet;

						("instrument_"++(1)++"_array_2").asSymbol.envirGet.size;
						("instrument_"++(2)++"_array_2").asSymbol.envirGet.size;
						("instrument_"++(3)++"_array_2").asSymbol.envirGet.size;
						*/

					}
					{~composition_process_setting == 2}{

						thisThread.randSeed = ~randseed_2;

						~part_bit_size = 64;
						~max_number_of_parts = 200; //max 200 parts. random probability choice decreases linearly from "~max_number_of_parts" value to 1.
						~number_of_parts_in_piece = (~max_number_of_parts.linrand - ~max_number_of_parts).neg;
						~just_ml_piece_size = ~part_bit_size*~number_of_parts_in_piece;

						thisThread.randSeed = ~randseed_3;

						~number_of_instruments.do(y = 0; {("dna_clock_"++(y=y+1)).asSymbol.envirPut(0);});


						//generation of piece

						~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_if_zero").asSymbol.envirPut("false");});

						~number_of_instruments.do(x = 0; {
							x=x+1;
							("instrument_"++(x)++"_array").asSymbol.envirPut(
								("instrument_"++(x)++"_array").asSymbol.envirGet.add((("instrument_"++(x)++"_freq").asSymbol.envirPut(~freqmap2.choose)).value)
							);
							("dna_clock_"++(x)++"_choice").asSymbol.envirPut(~dna_choice_array.choose);
						});

						~number_of_instruments.do(s = 0; {("old_instrument_"++(s=s+1)++"_freq").asSymbol.envirPut(("instrument_"++(s)++"_array").asSymbol.envirGet.at(0).asInteger + 12)});

						/*
						("dna_clock_"++(3)++"_choice").asSymbol.envirGet*/

						/*(~number_of_instruments-1).do(x = 1; {
						("instrument_"++(x=x+1)++"_freq").asSymbol.envirPut(0);
						("dna_clock_"++(x)++"_choice").asSymbol.envirPut(0);/*
						("dna_clock_"++(2)++"_choice").asSymbol.envirGet <~time_array.at(0)           (0);*/
						});*/

						/*(~number_of_instruments-1).do(s = 0; {("dna_clock_"++(s=s+1)).asSymbol.envirPut(("dna_clock_"++(s)).asSymbol.envirGet+1);});
						(~number_of_instruments-1).do(s = 1; {
						s=s+1;
						~dna_clock = ("dna_clock_"++(s)).asSymbol.envirGet;
						~inst_num = s;
						~inst_clock_func.value;
						});*/

						(~just_ml_piece_size-1).do({~inst_clock_func.value;});

						~number_of_instruments.do(n = 0; {
							n=n+1;
							("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_array").asSymbol.envirGet;);
							("instrument_"++(n)++"_array").asSymbol.envirPut([]);
							("instrument_"++(n)++"_full").asSymbol.envirPut(("instrument_"++(n)++"_full").asSymbol.envirGet.flatten);
						});
					};



					" 3".postln;
					//create pseq arrays function

					(
						~create_pseq_arrays_func = {

							h = 0.midicps;

							1.do({
								/*
								~number_of_instruments.do(x = 0; {("instrument_"++(x=x+1)++"_full").asSymbol.envirPut([]);});*/

								/*
								("instrument_"++(1)++"_full").asSymbol.envirGet.at(5800).size
								*/

								if(~saved_piece_loaded != 1, {

									~number_of_instruments.do(x = 0; {
										x=x+1;
										/*~number_of_parts_in_piece.do(y = 0; {("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.add(("instrument_"++(x)++"_array_part_"++(y=y+1)).asSymbol.envirGet););});
										*/("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.flatten);
										("instrument_"++(x)++"_full_midi").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet);
										("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.midicps);
										("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.replace(h, 0));
									});
								}, {
									~number_of_instruments.do(x = 0; {
										x=x+1;
										("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full_midi").asSymbol.envirGet.midicps);
										("instrument_"++(x)++"_full").asSymbol.envirPut(("instrument_"++(x)++"_full").asSymbol.envirGet.replace(h, 0));
									});
									~saved_piece_loaded = 0;
								});


								"Piece Creation Is Finsihed".postln;

								if(~piece_created == nil, {~piece_created = "true";});
								AppClock.sched(0, {
									~playbutton_func.value;
									~piecegenerated.string = "Piece Generated";
									AppClock.sched(3, {~piecegenerated.close});
								});

								AppClock.sched(0, {
									~fullpiecedur1 = ((~piecedur_secs = (("instrument_"++(1)++"_full").asSymbol.envirGet.size*~bit_duration).round)/60);
									~dd_secs = ((~fullpiecedur1-(~dd_mins = ~fullpiecedur1.roundUp-1))*60).roundUp;
									/*~fullpiecedur = ("piece length - "++(~dd_mins.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asString)++" mins");
									*/
									~fullpiecedur = ("Piece Length - "++(~piecedur_secs.asInteger.asString)++" secs ("++(~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString)++" mins)");
									piecedurationstatictext.string = ~fullpiecedur;
									("\npiece duration: \n"++((~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString))++" minutes";).postln;
									/*rangeslider.set(spec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"), initVal: [0,if(~piecedur_secs != nil, {~piecedur_secs}, {0})]);
									*/timeslider.set(spec: ControlSpec(0, if(~piecedur_secs != nil, {~piecedur_secs}, {0}), \lin, 1, 50, "secs"));
									~rate_selection_start = 0;
									~rate_selection_end = ("instrument_"++(1)++"_full").asSymbol.envirGet.size;
								});

								if(~quicknewpiecebutton == 1, {
									AppClock.sched(0, {
										~quicknewpiecebutton = 0;
										~stopbuttonfunc.value;
										~playbutton.value = 1;
										~run_piece_func.value;
									});
								});
							});
						};

						~create_pseq_arrays_func.value;
					);
				};

				~create_piece_function.value;

			);

			/*~create_piece_routine.isPlaying*/

		);
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// MACHINE LEARNING FUNCTION

	~machine_learning_func = {var interv_inst_case, time_inst_case;

		if(~pause_piece == 0, {~playbutton.valueAction = 0});

		~interval_array = (1..12);
		/*~dna_choice_array = (1..64);*/
		~time_array = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8];
		~dna_choice_array = [1, 2, 4, 8, 16, 32, 64, 128];
		/*~dna_choice_array = [~bit_duration*1, ~bit_duration*2, ~bit_duration*4, ~bit_duration*8, ~bit_duration*16, ~bit_duration*32, ~bit_duration*64];*/
		/*~time_array = [0.25, 1, 2, 4];*/
		~dna_choice_array_num = (1..~dna_choice_array.size);
		~net_change = 0.048;

		/*
		x=0; 128.do({("instrument_"++(1)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x=x+1)).postln});
		x=0; 128.do({("instrument_"++(2)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x=x+1)).postln})
		x=0; 128.do({("instrument_"++(3)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x=x+1)).postln})

		("instrument_"++(1)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1)).postln;
		("instrument_"++(2)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1)).postln;
		("instrument_"++(3)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1)).postln;*/

		/*("instrument_"++(1)++"_full_midi").asSymbol.envirPut(
		[56,56,56,56,56,56,56,56,56,56,76,76,76,76,76,76,0,0,0,0,0,0,0,0,66,66,66,66,66,66,66,66,66,66,66,66,66,66,66]
		);
		("instrument_"++(2)++"_full_midi").asSymbol.envirPut(
		[45,45,45,45,45,45,45,0,0,0,0,0,0,0,0,0,0,68,68,68,68,68,68,68,68,68,55,55,55,55,75,75,75,75,75,75,75,75]
		);
		("instrument_"++(3)++"_full_midi").asSymbol.envirPut(
		[87,87,87,87,87,87,87,87,87,96,96,96,96,96,96,0,0,0,0,0,0,79,79,79,79,75,75,75,75,75,75,75,71,71,71,71,71,71]
		);
		~number_of_instruments = 3;
		~rate_selection_start = 0;
		~rate_selection_end = ("instrument_"++(1)++"_full_midi").asSymbol.envirGet - 1;
		~inst_array = [1,2,3];*/

		/*~machine_learning_func.value;*/


		// create back up func. write to 2 different folders.


		~savenum = 0;

		interv_inst_case = {
			case
			{~number_of_instruments == 2}{(~interval_diff0.asString)++"__"++(~interval_diff2.asString)}
			{~number_of_instruments == 3}{(~interval_diff0.asString)++"__"++(~interval_diff2.asString)++"_"++(~interval_diff3.asString)};
		};

		(

			("\n").post;
			("intervals written -").post;
			("\n").post;

			~number_of_instruments.do(r = 0; {var freqmidinum;
				r=r+1;
				~inst_num = r;
				(~rate_selection_end-~rate_selection_start).do(x = 0; {
					x=x+1;
					j = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1));
					v = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+x);
					if(((v != j) and: (v != 0) and: (j != 0) and: (j != nil)), {freqmidinum = v;

						u = 0;

						// find interval of "older" notes
						q = x;
						c = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
						b = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-2));
						while({c == b}, {
							q=q-1;
							c = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
							b = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-2));
						});
						if(((b != 0) and: (b != nil)), {
							e = c - b;
							if(e.isNegative == true, {e = e.neg});
							while({e > 12}, {e = e - 12});
							("interval_diff"++(u)).asSymbol.envirPut(e.value;);
						}, {
							e = 12;
							("interval_diff"++(u)).asSymbol.envirPut(e.value;);

						});

						// find interval of "newer" notes
						e = v - j;
						if(e.isNegative == true, {e = e.neg});
						while({e > 12}, {e = e - 12});
						("interval_diff"++(u=u+1)).asSymbol.envirPut(e.value;);

						// find interval between intruments
						(~number_of_instruments-1).do(t = 1; {
							e = j - ("instrument_"++(~inst_array.wrapAt(~inst_num-(t=t-1)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1));
							if(e.isNegative == true, {e = e.neg});
							while({e > 12}, {e = e - 12});
							("interval_diff"++(u=u+1)).asSymbol.envirPut(e.value;);
						});

						(~number_of_instruments+1).do(u = (-1); {
							u=u+1;
							("interval_diff"++(u)).asSymbol.envirPut(("interval_diff"++(u)).asSymbol.envirGet.round.asInteger;);
						});

						/*~interval_diff0
						~interval_diff1
						~interval_diff2
						~interval_diff3*/
						/*
						((r.asString)++"-"++(freqmidinum.asString)++"-"++(~interval_diff1)++" ").post;*/

						((~interval_diff1-1).asString++" - "++(~player_array.wrapAt(r-1))++"__"++(interv_inst_case.value)).postln;

						/*("/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(~player_array.wrapAt(r-1))++"_interval_weights/"++(interv_inst_case.value)++"_weights.scd").asString.postln;
						*/
						h = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(~player_array.wrapAt(r-1))++"_interval_weights/"++(interv_inst_case.value)++"_weights.scd", false, true);
						h = h.flatten.asFloat;
						/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
						~interval_diff_revert_1.postln;*/
						h.put(~interval_diff1-1, (h.at(~interval_diff1-1) + (~net_change*~net_change_mult)));
						h = h.normalizeSum;
						g = File(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(~player_array.wrapAt(r-1))++"_interval_weights/"++(interv_inst_case.value)++"_weights.scd", "w");
						~interval_array.size.do(s = (-1); {g.write(h.at(s=s+1).asString++" ");});
						g.close;
					});
				});
			});
		);
		~savenum_interval = ~savenum;






		/*(~net_change*24);
		y=0.083333333333333;
		[ y=y+(~net_change*8), 0.083333333333333, 0.083333333333333, 0.083333333333333,
		0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333,
		0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333 ].normalizeSum;

		*/




		/*	((("inst_"++(w)).asSymbol.envirGet)++"_"++(0)++(0)++(0)++"_weights").asSymbol.envirGet



		("instrument_"++(1)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x)).postln

		("instrument_"++(1)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1)).postln;
		("instrument_"++(2)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1)).postln;
		("instrument_"++(3)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1)).postln;*/


		// do it where it counts number of bits per note... for each inst... then scan last note of same inst, as well as the notes
		// of other inst at time of start of new note for current inst

		/*("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(y=y-1);

		("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(y=y+1);*/

		~savenum = 0;

		time_inst_case = {
			case
			{~number_of_instruments == 2}{(~lengthplace2.asString)++"_"++(~lengthplace3.asString)}
			{~number_of_instruments == 3}{(~lengthplace2.asString)++"_"++(~lengthplace3.asString)++"_"++(~lengthplace4.asString)};
		};

		(
			("\n").post;
			("bit durations written -").post;
			("\n").post;

			~number_of_instruments.do(r = 0; {var freqmidinum;
				r=r+1;
				~inst_num = r;
				(~rate_selection_end-~rate_selection_start).do(x = 0; {
					x=x+1;
					j = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1));
					v = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+x);
					if(v != j, {freqmidinum = v;
						k = 0; c = (~rate_selection_start+(x)); u = 0;
						while({
							("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(c=c-1) ==
							("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1))
						}, {k = k +1});
						~length2 = k;
						k = 0; c = (~rate_selection_start+(x));
						while({
							("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(c=c+1) ==
							("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(c-1)
						}, {k = k +1});
						~length1 = k+1;
						(~number_of_instruments-1).do(t = 0; u = 2; {
							q = x-1;
							d = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
							f = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+q);
							while({f == d}, {
								q=q-1;
								d = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
								f = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+q);
							});
							c = (~rate_selection_start+(q+1));
							k = 0;
							while({
								("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(c=c+1) ==
								("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(c-1)
							}, {k = k +1});
							k=k+2;
							("length"++(u=u+1)).asSymbol.envirPut(k);
							t=t-1;
						});

						/*~length1.postln;*/
						/*~length1
						~length2
						~length3
						~length4
						*/
						/*~interval_diff1
						~interval_diff2
						~interval_diff3*/

						(~number_of_instruments+1).do(u = 0; {
							u=u+1;
							("length"++(u)).asSymbol.envirPut(("length"++(u)).asSymbol.envirGet.round.asInteger;);
						});

						(~number_of_instruments+1).do(u = 0; {
							u=u+1;
							case
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(0)) and: (("length"++(u)).asSymbol.envirGet*~bit_duration <= ~time_array.at(1))}{if((("length"++(u)).asSymbol.envirGet*~bit_duration < (~time_array.at(0)+~time_array.at(1)/2)), {("lengthplace"++(u)).asSymbol.envirPut(0)}, {("lengthplace"++(u)).asSymbol.envirPut(1)})}
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(1)) and: (("length"++(u)).asSymbol.envirGet*~bit_duration <= ~time_array.at(2))}{if((("length"++(u)).asSymbol.envirGet*~bit_duration < (~time_array.at(1)+~time_array.at(2)/2)), {("lengthplace"++(u)).asSymbol.envirPut(1)}, {("lengthplace"++(u)).asSymbol.envirPut(2)})}
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(2)) and: (("length"++(u)).asSymbol.envirGet*~bit_duration <= ~time_array.at(3))}{if((("length"++(u)).asSymbol.envirGet*~bit_duration < (~time_array.at(2)+~time_array.at(3)/2)), {("lengthplace"++(u)).asSymbol.envirPut(2)}, {("lengthplace"++(u)).asSymbol.envirPut(3)})}
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(3)) and: (("length"++(u)).asSymbol.envirGet*~bit_duration <= ~time_array.at(4))}{if((("length"++(u)).asSymbol.envirGet*~bit_duration < (~time_array.at(3)+~time_array.at(4)/2)), {("lengthplace"++(u)).asSymbol.envirPut(3)}, {("lengthplace"++(u)).asSymbol.envirPut(4)})}
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(4)) and: (("length"++(u)).asSymbol.envirGet*~bit_duration <= ~time_array.at(5))}{if((("length"++(u)).asSymbol.envirGet*~bit_duration < (~time_array.at(4)+~time_array.at(5)/2)), {("lengthplace"++(u)).asSymbol.envirPut(4)}, {("lengthplace"++(u)).asSymbol.envirPut(5)})}
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(5)) and: (("length"++(u)).asSymbol.envirGet*~bit_duration <= ~time_array.at(6))}{if((("length"++(u)).asSymbol.envirGet*~bit_duration < (~time_array.at(5)+~time_array.at(6)/2)), {("lengthplace"++(u)).asSymbol.envirPut(5)}, {("lengthplace"++(u)).asSymbol.envirPut(6)})}
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(6)) and: (("length"++(u)).asSymbol.envirGet*~bit_duration <= ~time_array.at(7))}{if((("length"++(u)).asSymbol.envirGet*~bit_duration < (~time_array.at(6)+~time_array.at(7)/2)), {("lengthplace"++(u)).asSymbol.envirPut(6)}, {("lengthplace"++(u)).asSymbol.envirPut(7)})}
							{(("length"++(u)).asSymbol.envirGet*~bit_duration >= ~time_array.at(7))}{("lengthplace"++(u)).asSymbol.envirPut(7)};

						});

						(~number_of_instruments+1).do(u = 0; {
							u=u+1;
							("lengthplace"++(u)).asSymbol.envirPut(("lengthplace"++(u)).asSymbol.envirGet.round.asInteger;);
						});

						/*~interval_diff3*~bit_duration

						~length1
						~length2
						~length3
						~length4
						~lengthplace2
						~lengthplace3
						~lengthplace4*/
						/*"write".postln;*/

						// don't think ~lengthplace matters with players. doesn't work like that in inst_clock_func.
						/*
						((r.asString)++"-"++(freqmidinum.asString)++"-"++(~length1)++" ").post;*/

						((~lengthplace1).asString++" - "++(~player_array.wrapAt(r-1))++" "++(time_inst_case.value)).postln;

						/*("/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(~player_array.wrapAt(r-1))++"_time_weights/"++(time_inst_case.value)++"_time_weights.scd").asString.postln;
						*/
						h = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(~player_array.wrapAt(r-1))++"_time_weights/"++(time_inst_case.value)++"_time_weights.scd", false, true);
						h = h.flatten.asFloat;
						("lengthplace_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++"__"++(time_inst_case.value), h]);
						h.put(~lengthplace1, (h.at(~lengthplace1) + (~net_change*~net_change_mult)));
						h = h.normalizeSum;
						g = File(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(~player_array.wrapAt(r-1))++"_time_weights/"++(time_inst_case.value)++"_time_weights.scd", "w");
						~dna_choice_array.size.do(s = (-1); {g.write(h.at(s=s+1).asString++" ");});
						g.close;

						~rate_revert = 0;
					});
				});
			});
		);
		~savenum_time = ~savenum;





		/*("player_"++(w)++"__"++(0)++"_"++(0)++"_"++(0)++"_time_weights").asSymbol.envirGet*/



		AppClock.sched(0.25, {
			if(~pause_piece == 1, {~playbutton.valueAction = 1});

			if(ratedonetext != nil, {ratedonetext.close; ratedonetext = nil;});
			ratedonetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.063829787234, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/4.04545454545, ~firstviewboundsHEIGHT/21));
			ratedonetext.background=~backgroundcolor;
			ratedonetext.stringColor_(~stringcolor);
			ratedonetext.string = "PIECE SELECTION RATED";
			ratedonetext.font = Font(size: ~firstviewboundsWIDTH/55);

			AppClock.sched(3, {if(ratedonetext != nil, {ratedonetext.close; ratedonetext = nil;});});
		});

	};

	~ml_revert_func = {

		/*~randseed_1_backup2 = ~randseed_1;
		~randseed_2_backup2 = ~randseed_2;
		~randseed_3_backup2 = ~randseed_3;
		~randseed_1 =  ~randseed_1_backup;
		~randseed_2 =  ~randseed_2_backup;
		~randseed_3 =  ~randseed_3_backup;
		~randseed_1_backup = ~randseed_1_backup2;
		~randseed_2_backup = ~randseed_2_backup2;
		~randseed_3_backup = ~randseed_3_backup2;*/

		if(~duranduranbot_routine.isPlaying == true, {if(~pause_piece == 0, {~playbutton.valueAction = 0});});

		/*~savenum_interval.do(r = 0; {
		r = r+1;
		h = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0))++"_weights.scd", false, true);
		h = h.flatten.asFloat;
		("interval_diff_revert2_"++(r)).asSymbol.envirPut([(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0)), h]);
		});
		~savenum_time.do(r = 0; {
		r = r+1;
		h = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0))++"_time_weights.scd", false, true);
		h = h.flatten.asFloat;
		("lengthplace_revert2_"++(r)).asSymbol.envirPut([(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0)), h]);
		});

		~savenum_interval.do(r = 0; {
		r = r+1;
		h = ("interval_diff_revert_"++(r)).asSymbol.envirGet.at(1);
		g = File(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0))++"_weights.scd", "w");
		~interval_array.size.do(s = (-1); {g.write(h.at(s=s+1).asString++" ");});
		g.close;
		});
		~savenum_time.do(r = 0; {
		r = r+1;
		h = ("lengthplace_revert_"++(r)).asSymbol.envirGet.at(1);
		g = File(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0))++"_time_weights.scd", "w");
		~dna_choice_array.size.do(s = (-1); {g.write(h.at(s=s+1).asString++" ");});
		g.close;
		});

		~savenum_interval.do(r = 0; {
		r = r+1;
		h = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_interval_weights/"++(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0))++"_weights.scd", false, true);
		h = h.flatten.asFloat;
		("interval_diff_revert_"++(r)).asSymbol.envirPut([(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0)), (("interval_diff_revert2_"++(r)).asSymbol.envirGet.at(1))]);
		});
		~savenum_time.do(r = 0; {
		r = r+1;
		h = FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/ddbot_training_data_current_"++(~number_of_instruments.asString)++"_inst/ddbot_time_weights/"++(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0))++"_time_weights.scd", false, true);
		h = h.flatten.asFloat;
		("lengthplace_revert_"++(r)).asSymbol.envirPut([(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0)), (("lengthplace_revert2_"++(r)).asSymbol.envirGet.at(1))]);
		});

		if(~rate_revert == 0, {"Rates Reverted".postln; ~rate_revert = 1;}, {"Rates Reverted To Last Rated".postln; ~rate_revert = 0;});
		*/
		AppClock.sched(0.25, {
			if(~duranduranbot_routine.isPlaying == true, {if(~pause_piece == 1, {~playbutton.valueAction = 1});});

			if(ratedonetext != nil, {ratedonetext.close; ratedonetext = nil;});
			ratedonetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.063829787234, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/4.04545454545, ~firstviewboundsHEIGHT/21));
			ratedonetext.background=~stringcolor;
			ratedonetext.stringColor_(~backgroundcolor);
			ratedonetext.string = "RATES REVERTED";
			ratedonetext.font = Font(size: ~firstviewboundsWIDTH/55);

			AppClock.sched(3, {if(ratedonetext != nil, {ratedonetext.close; ratedonetext = nil;});});
		});
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// RUN/PLAY PIECE


	/*	~klangarray1 = {Array.rand(3, 0, 20);};

	~number_of_instruments.do(x = 0; {
	SynthDef("synth"++(x=x+1), {arg freq = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
	var sig;
	sig = Mix(DynKlang.ar(`[ [freq]++~klangarray1.value+freq, nil, nil ], 1, 0) * 0.05);
	sig = RLPF.ar(sig, ffreq, rq);
	sig = sig * EnvGen.kr(
	Env.adsr(0.01, 0.1, 0.6, 0.1),
	gt
	);

	Out.ar(0, sig!2);
	}).play;
	});*/

	if(~buffer == nil, {~buffer = Buffer.read(s, Document.current.dir++"/other_program_files/samples/Alesis-Sanctuary-QCard-Promenade-C3-loop5.wav");});

	~vd_midi = 73; //base note for audio file - for equalizing volume/amplitude. default is 73.

	~number_of_instruments.do(x = 0; {
		SynthDef("synth"++(x=x+1), {arg out = 0, bufnum = 0, freq = 1, amp = 1;
			var sig;
			sig = PlayBuf.ar(1, ~buffer, freq/~vd_midi.midicps, startPos: 0, loop: 1);
			sig = BBandPass.ar(sig, freq: freq, bw: 1.6, mul: 1, add: 0);
			/*sig = BPF.ar(sig, freq: freq, rq: 1.3, mul: 1, add: 0);*/
			/*sig = BLowPass.ar(sig, freq: freq, rq: 4, mul: 1, add: 0);*/
			/*sig = BLowPass4.ar(sig, freq: freq, rq: 6, mul: 1, add: 0);*/
			/*sig = LPZ2.ar(sig, mul: 1, add: 0);*/
			/*sig = BPZ2.ar(sig, mul: 1, add: 0);*/

			Out.ar(0, ((sig.dup)*amp));
		}).add;
	});

	/*{ SinOsc.ar(73.midicps, 0, 0.5) }.play;*/

	/*SynthDef("synth1", {| out = 0, bufnum = 0, freq = 1 |
	var sig;
	sig = PlayBuf.ar(1, ~buffer, freq/73.midicps, startPos: 0, loop: 1);
	Out.ar(1, sig!2);
	}).play;*/



	/*~piecedur_secs.postln;
	~number_of_instruments.do(x = 0; {
	SynthDef("synth"++(x=x+1), {
	arg freq = 500, mRatio = 1, cRatio = 1, index = 1, iScale = 5, amp = 0.2, atk = 0.01, rel = 300, cAtk = 4, cRel = (-4), pan = 0;
	var car, mod, env, iEnv;
	iEnv = EnvGen.kr(
	Env.new(
	[index, index * iScale, index],
	[atk, rel],
	[cAtk, cRel]
	)
	);
	env = EnvGen.kr(
	Env.perc(atk, rel, curve: [cAtk, cRel])/*,
	doneAction: 2*/
	);
	mod = SinOsc.ar(freq * mRatio, mul: freq * mRatio * iEnv);
	car = SinOsc.ar(freq * cRatio + mod) * env * amp;
	car = Pan2.ar(car, pan);
	Out.ar(0, car);
	}).add;
	});*/


	~run_piece_func = {

		(


			/*SynthDef(\bass_synth1, {arg freq = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
			var sig;
			sig = Mix(DynKlang.ar(`[ [freq]++~klangarray1+freq, nil, nil ], 1, 0) * 0.05);
			sig = RLPF.ar(sig, ffreq, rq);
			sig = sig * EnvGen.kr(
			Env.adsr(0.01, 0.1, 0.6, 0.1),
			gt
			);

			Out.ar(0, sig!2);
			}).add;
			SynthDef(\keys_synth2, {arg freqa = 432, freqb = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
			var sig;
			sig = Mix([DynKlang.ar(`[ [freqa]++~klangarray2+freqa, nil, nil ], 1, 0), DynKlang.ar(`[ [freqb]++~klangarray2+freqb, nil, nil ], 1, 0)]  * 0.05);
			sig = RLPF.ar(sig, ffreq, rq);
			sig = sig * EnvGen.kr(
			Env.adsr(0.01, 0.1, 0.6, 0.1),
			gt
			);

			Out.ar(0, sig!2);
			}).add;

			/*Synth(\keys_synth2, [\freqa, 432, \freqb, 300])*/

			SynthDef(\vocal_synth3, {arg freq = 432, freq1 = 60, amp = 0.1, detun = 1.006, ffreq = 800, rq = 1, gt = 1;
			var sig;
			sig = Mix(DynKlang.ar(`[ [freq]++~klangarray3+freq, nil, nil ], 1, 0) * 0.05);
			sig = RLPF.ar(sig, ffreq, rq);
			sig = sig * EnvGen.kr(
			Env.adsr(0.01, 0.1, 0.6, 0.1),
			gt
			);

			Out.ar(0, sig!2);
			}).add;*/

			/*b = Buffer.read(s, Document.current.dir++"/samples/Alesis-Sanctuary-QCard-Promenade-C3-loop5.wav");

			~synth = {
			SynthDef(\help_PlayBuf, {| out = 0, bufnum = 0, freq = 1 |
			Out.ar(out,
			(PlayBuf.ar(1, b, BufRateScale.kr(bufnum)*freq, startPos: 0, loop: 1))!2
			)
			}).play(s, [\out, 0]);
			};

			~synth = {
			SynthDef(\help_PlayBuf, {| out = 0, bufnum = 0, s1,s2,s3,s4, freq1 = 1, freq2 = 1.3, freq3 = 1.5 |
			s1 = PlayBuf.ar(1, b, BufRateScale.kr(bufnum)*freq1, startPos: 0, loop: 1);
			s2 = PlayBuf.ar(1, b, BufRateScale.kr(bufnum)*freq2, startPos: 0, loop: 1);
			s3 = PlayBuf.ar(1, b, BufRateScale.kr(bufnum)*freq3, startPos: 0, loop: 1);
			s4 = Mix.new([s1, s2, s3]);
			Out.ar(out, s4!2);
			}).play(s, [\out, 0]);
			};*/

			/*
			~duranduranbotsynth1 = ~synth.value;
			~duranduranbotsynth2 = ~synth.value;
			~duranduranbotsynth3 = ~synth.value;

			~duranduranbotsynth1.set(\freq, 2);
			*/

			/*~timer_routine = Task(
			~dd_secs = ((~fullpiecedur1-(~dd_mins = ~fullpiecedur1.roundUp-1))*60).roundUp;
			~piecedur_secs = (~dd_mins*60)+~dd_secs;
			~piecedur_secs_zero = 0;
			{
			~piecedur_secs.do({
			AppClock.sched(0, {
			piecedurationstatictext.string = ("piece length - "++(~piecedur_secs_zero.asString)++" secs ("++(~dd_mins.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asString)++" mins)");
			~piecedur_secs_zero = ~piecedur_secs_zero+1;
			~dd_secs = ~dd_secs-1;
			if(~dd_secs == (-1), {~dd_mins = ~dd_mins-1; ~dd_secs = 59;});

			/*if(~piecedur_secs_zero >= 20, {rangeslider.valueAction = [~piecedur_secs_zero-20, ~piecedur_secs_zero]}, {rangeslider.valueAction = [0, ~piecedur_secs_zero]});
			*/timeslider.valueAction = ~piecedur_secs_zero;
			});

			1.wait;
			});
			});*/

			~init_start_func = {

				Server.default.makeBundle(Server.default.latency, {

					~number_of_instruments.do(~x = 0; {("duranduranbotsynth"++(~x=~x+1)).asSymbol.envirGet.set(
						\amp, (~amp - ((("instrument_"++(~x)++"_full_midi").asSymbol.envirGet.at(~piece_position) - ~vd_midi)/~vd_midi))
					);
					});

					~number_of_instruments.do(~x = 0; {("duranduranbotsynth"++(~x=~x+1)).asSymbol.envirPut(
						Synth("synth"++(~x), [freq: ("instrument_"++(~x)++"_full").asSymbol.envirGet.at(~piece_position)]);

					);});


					/*~duranduranbotsynth1_m = Synth(
					(
					case
					{~inst_1 == "bass"}{\synth1}
					{~inst_1 == "keys"}{\synth2}
					{~inst_1 == "vocal"}{\synth3};
					), [freq: ~instrument_1_full_mirrored.at(i)]);
					~duranduranbotsynth2_m = Synth(
					(
					case
					{~inst_2 == "bass"}{\synth1}
					{~inst_2 == "keys"}{\synth2}
					{~inst_2 == "vocal"}{\synth3};
					), [freq: ~instrument_2_full_mirrored.at(i)]);
					~duranduranbotsynth3_m = Synth(
					(
					case
					{~inst_3 == "bass"}{\synth1}
					{~inst_3 == "keys"}{\synth2}
					{~inst_3 == "vocal"}{\synth3};
					), [freq: ~instrument_3_full_mirrored.at(i)]);*/


					/*~duranduranbotsynth1 = ~synth.value;
					~duranduranbotsynth2 = ~synth.value;
					~duranduranbotsynth3 = ~synth.value;
					~duranduranbotsynth1.set(\freq, ~instrument_1_full.at(x));
					~duranduranbotsynth2.set(\freq, ~instrument_2_full.at(x));
					~duranduranbotsynth3.set(\freq, ~instrument_3_full.at(x));*/
				});
			};

			~init_func = {

				Server.default.makeBundle(Server.default.latency, {

					~number_of_instruments.do(~x = 0; {("duranduranbotsynth"++(~x=~x+1)).asSymbol.envirGet.set(
						\amp, (~amp - ((("instrument_"++(~x)++"_full_midi").asSymbol.envirGet.at(~piece_position) - ~vd_midi)/~vd_midi))
					);
					});
					~number_of_instruments.do(~x = 0; {("duranduranbotsynth"++(~x=~x+1)).asSymbol.envirGet.set(
						\freq, ("instrument_"++(~x)++"_full").asSymbol.envirGet.at(~piece_position);

					);});

					/*~duranduranbotsynth1_m.set(\freq, ~instrument_1_full_mirrored.at(i));
					~duranduranbotsynth2_m.set(\freq, ~instrument_2_full_mirrored.at(i));
					~duranduranbotsynth3_m.set(\freq, ~instrument_3_full_mirrored.at(i));
					*/
				});

			};

			case
			{~loop_on == 0}{

				~duranduranbot_routine = Task({var bit_timer;

					bit_timer = 0;
					~dd_secs = ((~fullpiecedur1-(~dd_mins = ~fullpiecedur1.roundUp-1))*60).roundUp;
					~piecedur_secs = (~dd_mins*60)+~dd_secs;
					~piecedur_secs_zero = 0;
					~bit_dur_rec = ~bit_duration.reciprocal;

					~bit_timer_func = {
						~rate_selection_end = ~piece_position;
						~rate_selection_start = ~piece_position-(~auto_select_diff/~bit_duration);
						if(~rate_selection_start.isNegative == true, {~rate_selection_start = 0;});

						AppClock.sched(0, {
							piecepositionstatictext.string = (
								"Bit pos. "++(~piece_position.asInteger.asString)++" Midi "++(
									case
									{~number_of_instruments == 2}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)}
									{~number_of_instruments == 3}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)++" "++(~instrument_3_full_midi.at(~piece_position).asString)};
								)
							);
						});

						bit_timer = bit_timer + 1;
						if(bit_timer >= ~bit_dur_rec, {
							AppClock.sched(0, {
								timeslider.valueAction = (~piecedur_secs_zero = timeslider.value+1);
								~dd_secs = ~dd_secs-1;
								if(~dd_secs == (-1), {~dd_mins = ~dd_mins-1; ~dd_secs = 59;});
								piecedurationstatictext.string = ("Piece Length - "++(~piecedur_secs_zero.asInteger.asString)++" secs ("++(~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString)++" mins)");
								/*if(timeslider.value >= ~auto_select_diff, {rangeslider.valueAction = [timeslider.value-~auto_select_diff, timeslider.value]}, {rangeslider.valueAction = [0, timeslider.value]});
								*//*if(~piecedur_secs_zero >= (~piecedur_secs), {~stop_func.value; "piece finished".postln; piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"piece length - --:-- mins"});})
								*//*if(~piece_position == ~instrument_1_full.size, {
								AppClock.sched(~bit_duration*4, {
								~stop_func.value; "piece finished".postln; piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"Piece Length - --:-- mins"});
								});
								});*/
							});
							bit_timer = 0;
						});
					};

					~piece_position = 0;

					1.do({

						if(~will_record == 1, {
							thisProcess.platform.recordingsDir = (Document.current.dir++"/other_stuff_(recordings_etc)/duranduranbot_recordings");
							Server.default.record;
							1.wait;
						});

						"piece currently running".postln;

						~init_start_func.value;

						if(~will_record != 1, {~bit_timer_func.value;});

						~bit_duration.wait;

					});

					inf.do({

						~piece_position = ~piece_position + 1;

						~init_func.value;

						if(~will_record != 1, {~bit_timer_func.value;});

						/*t=170;
						~instrument_2_full_midi.at(t=t-1)*/


						~bit_duration.wait;
						/*
						~piece_position.postln;*/

						//
						// ("pp "++~piece_position.value++" bit_timer "++bit_timer.value).postln;

						/*~piece_position.postln;*/

						if(~piece_position >= (~instrument_1_full.size-1), {
							var amp = ~amp;
							100.do({~number_of_instruments.do({

								~amp = ~amp - (~amp/25);
								Server.default.makeBundle(Server.default.latency, {
									~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(
										\amp, ~amp
									);});
								});
								0.01.wait;});});
							Server.default.makeBundle(Server.default.latency, {
								~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(
									\amp, 0
								);});
							});
							~amp = 1;
							1.wait; if(~will_record == 1, {Server.default.stopRecording;}); ~stop_func.value; "piece finished".postln;
							if((FileReader.read(Document.current.dir++"/other_program_files/autonewpiece.scd", false, true).flatten.at(0) == 1.asString), {AppClock.sched(0, {quicknewpiecebutton.valueAction = 0;});});
						});

					});
				});

			}
			{~loop_on == 1}{

				if(~will_record == 1, {
					thisProcess.platform.recordingsDir = (Document.current.dir++"/other_stuff_(recordings_etc)/duranduranbot_recordings");
					Server.default.record;
				});

				~duranduranbot_routine = Task({var bit_timer, numberofloops = 0;

					bit_timer = 0;
					~dd_secs = ((~fullpiecedur1-(~dd_mins = ~fullpiecedur1.roundUp-1))*60).roundUp;
					~piecedur_secs = (~dd_mins*60)+~dd_secs;
					~piecedur_secs_zero = 0;
					~bit_dur_rec = ~bit_duration.reciprocal;

					~bit_timer_func = {
						~rate_selection_end = ~piece_position;
						~rate_selection_start = ~piece_position-(~auto_select_diff/~bit_duration);
						if(~rate_selection_start.isNegative == true, {~rate_selection_start = 0;});

						AppClock.sched(0, {
							piecepositionstatictext.string = (
								"Bit pos. "++(~piece_position.asInteger.asString)++" Midi "++(
									case
									{~number_of_instruments == 2}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)}
									{~number_of_instruments == 3}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)++" "++(~instrument_3_full_midi.at(~piece_position).asString)};
								)
							);
						});

						bit_timer = bit_timer + 1;
						if(bit_timer >= ~bit_dur_rec, {
							AppClock.sched(0, {
								timeslider.valueAction = (~piecedur_secs_zero = timeslider.value+1);
								~dd_secs = ~dd_secs-1;
								if(~dd_secs == (-1), {~dd_mins = ~dd_mins-1; ~dd_secs = 59;});
								piecedurationstatictext.string = ("Piece Length - "++(~piecedur_secs_zero.asInteger.asString)++" secs ("++(~dd_mins.asInteger.asString)++":"++(if(~dd_secs<10,{"0"},{""})++~dd_secs.asInteger.asString)++" mins)");
								/*if(timeslider.value >= ~auto_select_diff, {rangeslider.valueAction = [timeslider.value-~auto_select_diff, timeslider.value]}, {rangeslider.valueAction = [0, timeslider.value]});
								*//*if(~piecedur_secs_zero >= (~piecedur_secs), {~stop_func.value; "piece finished".postln; piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"piece length - --:-- mins"});})
								*//*if(~piece_position == ~instrument_1_full.size, {
								AppClock.sched(~bit_duration*4, {
								~stop_func.value; "piece finished".postln; piecedurationstatictext.string = if(~fullpiecedur != nil, {~fullpiecedur}, {"Piece Length - --:-- mins"});
								});
								});*/
							});
							bit_timer = 0;
						});
					};

					~piece_position = ~pos_1;

					1.do({

						1.wait;

						~init_start_func.value;

						/*~bit_timer_func.value;*/

						~bit_duration.wait;

					});

					inf.do({

						~piece_position = ~piece_position + 1;

						~init_func.value;

						/*	~bit_timer_func.value;
						*/
						AppClock.sched(0, {
							piecepositionstatictext.string = (
								"Bit pos. "++(~piece_position.asInteger.asString)++" Midi "++(
									case
									{~number_of_instruments == 2}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)}
									{~number_of_instruments == 3}{(~instrument_1_full_midi.at(~piece_position).asString)++" "++(~instrument_2_full_midi.at(~piece_position).asString)++" "++(~instrument_3_full_midi.at(~piece_position).asString)};
								)
							);
						});

						/*t=170;
						~instrument_2_full_midi.at(t=t-1)*/

						if(~loop_on == 1, {if(~piece_position == ~pos_2;, {~piece_position = ~pos_1; numberofloops=numberofloops+1});});

						~bit_duration.wait;
						/*
						~piece_position.postln;*/

						//
						// ("pp "++~piece_position.value++" bit_timer "++bit_timer.value).postln;

						/*~piece_position.postln;*/

						if(numberofloops >= ~num_loops, {
							var amp = ~amp;
							100.do({~number_of_instruments.do({

								~amp = ~amp - (~amp/25);
								Server.default.makeBundle(Server.default.latency, {
									~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(
										\amp, ~amp
									);});
								});
								0.01.wait;});});
							Server.default.makeBundle(Server.default.latency, {
								~number_of_instruments.do(j = 0; {("duranduranbotsynth"++(j=j+1)).asSymbol.envirGet.set(
									\amp, 0
								);});
							});
							~amp = 1;
							1.wait; if(~will_record == 1, {Server.default.stopRecording;}); ~stop_func.value; "piece finished".postln;
							if((FileReader.read(Document.current.dir++"/other_program_files/autonewpiece.scd", false, true).flatten.at(0) == 1.asString), {AppClock.sched(0, {quicknewpiecebutton.valueAction = 0;});});
						});

					});

				});

			};


			Server.default.makeBundle(Server.default.latency, {
				/*	~timer_routine.start;*/
				~duranduranbot_routine.play;
			});

		);
	};







	w.front;

	~front_page_func.value;

};

if(FileReader.read(Document.current.dir++"/other_program_files/ddbot_training_data_folder/zdummyfile.scd", false, true) != nil,
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

/*~duranduranbot_GUI_func.value;*/








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