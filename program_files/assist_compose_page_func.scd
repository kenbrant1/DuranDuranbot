~assist_compose_page_func = {var fillertext,fillertext2;

	/*
	WHAT WE WANT

	- All functionality on the assist compose page - play button, rate, slider, etc
	- *everything* can be saved for a later date
	-
	*/

	~ai_assist_compose = 1;

	~firstview.close;

	/*~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;

	~secondview = ScrollView(~firstview, Rect(0, 200, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;*/

	/*~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;*/

	~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height/2.25));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height*2.25;

	~firstview2 = ScrollView(w, Rect(0, w.bounds.height/2.25, w.bounds.width, w.bounds.height - (w.bounds.height/2.25)));
	/*view.clearOnRefresh = false;*/
	~firstview2.background = ~backgroundcolor;
	~firstviewboundsWIDTH2 = ~firstview.bounds.width;
	~firstviewboundsHEIGHT2 = ~firstview.bounds.height*2.25;

	~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.75, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
	~firstviewtext1.background=~backgroundcolor;
	~firstviewtext1.stringColor_(~stringcolor);
	~firstviewtext1.string = "DDbot Assist Composition";
	~firstviewtext1.align = \topLeft;
	~firstviewtext1.font = Font(size: ~firstviewboundsWIDTH/48);

	~firstviewtext11 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21), ~firstviewboundsWIDTH/1.3, ~firstviewboundsHEIGHT/21));
	~firstviewtext11.background=~backgroundcolor;
	~firstviewtext11.stringColor_(~stringcolor);
	~firstviewtext11.string = "*this page is currently under development. See Instructions on what works.";
	~firstviewtext11.font = Font(size: ~firstviewboundsWIDTH/55);

	~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
        ~firstview.close;
        ~main_page_func.value;
    });
    ~firstviewtext2.background=~stringcolor;
    ~firstviewtext2.string = "Back";
    ~firstviewtext2.font = Font(size: ~firstviewboundsWIDTH/48);

	~generatepiecebutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/6.5454545454545, ~firstviewboundsHEIGHT/21)).states_([["Generate Piece",~stringcolor,~backgroundcolor],["Generate Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
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
	~generatepiecebutton.font = Font(size: ~firstviewboundsWIDTH/48);

	/*if(playnil == nil, {nil}, {*/
	/*~play_visible_func = {
	~play_visible_func_verify*/

	~playbutton_func = {
		~playbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/4.95, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Play Piece",~stringcolor,~backgroundcolor],["Pause Piece",~stringcolor,~backgroundcolor]]).action_({arg button;
			var pause_piece;
			if(button.value == 1, {
				if(~pause_piece != 1, {
					~seedscheck = 1;
					~stopbuttonfunc.value;
					~assisted_ai_run_piece_func.value;
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
				*/if(~ai_assist_compose != 1, {"Piece Paused".postln;});
				if(~will_record == 1, {s.pauseRecording;});
				~pause_piece = 1;
				~playbutton.string = "Resume";
			});

		});
		~playbutton.font = Font(size: ~firstviewboundsWIDTH/48);
		/*};*/
		~arrow = StaticText(~firstview, Rect(~firstviewboundsWIDTH/5.7016393442623, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/40, ~firstviewboundsHEIGHT/21));
		~arrow.background=~backgroundcolor;
		~arrow.stringColor_(~stringcolor);
		~arrow.string = "---";
		~arrow.font = Font(size: ~firstviewboundsWIDTH/48);
	};

	~playbutton_func.value;
	if(~pause_piece == 1, {~playbutton.string = "Resume"; ~stopbuttonfunc.value;});

	if(~piece_created == nil, {~playbutton.close; ~arrow.close;});

	~stopbuttonfunc = {~stopbutton = Button.new(~firstview, Rect(~firstviewboundsWIDTH/2.93, ~firstviewboundsHEIGHT/7.5, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Stop piece",~stringcolor,~backgroundcolor]]).action_({arg button;
		~stop_func.value;
	});

	~stopbutton.font = Font(size: ~firstviewboundsWIDTH/48);
	};

	if(~duranduranbot_routine.isPlaying == true, {~playbutton.value = 1; ~stopbuttonfunc.value}, {~playbutton.value = 0});


	~firstviewtext3 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Instructions",~stringcolor,~backgroundcolor],["Instructions",~stringcolor,~backgroundcolor]]).action_({arg button;
		~ai_assist_instructions_page_func.value;

	});
	~firstviewtext3.font = Font(size: ~firstviewboundsWIDTH/48);

	~save_assist_project_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Save Project",~stringcolor,~backgroundcolor]]).action_({arg button;

		"please wait".postln;
		g = File((~ddbot_root)++"/program_files/saved_projects/project_"++(z=((Date.getDate.format("%d_%m_%Y__%H_%M_%S").asString)))++".scd", "w");
		g.write("~num_assist_parts = "++(~num_assist_parts.asString)++";");
		~num_assist_parts.do(j = 0; {
			j=j+1;
			~number_of_instruments.do(s = 0; {
				s=s+1;
				g.write("\n~assisted_inst_"++(s)++"_array_part_"++(j)++" = [ ");
				("assisted_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirGet.size.do(q = (-1); {
					q=q+1;
					g.write((("assisted_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirGet.at(q).asString)++", ");
				});
				g.write("];");
			});
		});
		g.write("\n
if(~firstview != nil, {~firstview.close;});
if(~firstview2 != nil, {~firstview2.close;});
~assist_compose_page_func.value;
~firstview.refresh;
~firstview2.refresh;"++
			"project loaded".quote++".postln;
");
		g.close;
		"project saved".postln;
	});
	~save_assist_project_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~load_saved_assist_project_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21*2), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Load Saved Project",~stringcolor,~backgroundcolor]]).action_({arg button;

		FileDialog.new(
			{arg path;
				File.readAllString(path, r).interpret.value;
		}, {}, 0, 0, stripResult: true);
	});
	~load_saved_assist_project_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~new_assist_project_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/1.37, ~firstviewboundsHEIGHT/43.7+(~firstviewboundsHEIGHT/21*3), ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["New Proj. (will save curr.)",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~save_assist_project_button.valueAction = 0;
		~num_assist_parts.do(j = 0; {
			j=j+1;
			~number_of_instruments.do(s = 0; {
				s=s+1;
				("assisted_inst_"++(s)++"_array_part_"++(j)).asSymbol.envirPut(nil);
			});
		});
		~num_assist_parts = nil;
		~line_multiple1 = nil;
		~line_multiple2 = nil;
		if(~firstview != nil, {~firstview.close;});
		if(~firstview2 != nil, {~firstview2.close;});
		~assist_compose_page_func.value;
		~firstview.refresh;
		~firstview2.refresh;
	});
	~new_assist_project_button.font = Font(size: ~firstviewboundsWIDTH/48);


	if(~piece_position == nil, {~piece_position = 0});

	~timeslider = EZSlider(~firstview,  bounds: (Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/5, ~firstviewboundsWIDTH/1.5, ~firstviewboundsHEIGHT/21)), controlSpec: ControlSpec(0, ("instrument_"++(1)++"_full").asSymbol.envirGet.size, \lin, 1, 50, "bits"), action: {
		|timeselect|
		var selection1;
		~piece_position = timeselect.value/**(~instrument_1_full.size/~piecedur_secs)*/; // works
		/*selection1 = ~piece_position.roundUp;
		if(selection1 != ~piece_position, {~piece_position = selection1-1});*/
		/*if(timeslider.value >= ~rate_time_range, {rangeslider.valueAction = [timeslider.value-~rate_time_range, timeslider.value]}, {rangeslider.valueAction = [0, timeslider.value]});
		*//*timeselect.value.postln;
		~piece_position.postln;*/
		/*if((~piecedur_secs_zero+1) != timeselect.value, {
		~piecedur_secs_zero = timeselect.value;
		~fullpiecedur_change = ((~piecedur_secs - ~piecedur_secs_zero)/60);
		~dd_secs = ((~fullpiecedur_change-(~dd_mins = ~fullpiecedur_change.roundUp-1))*60).roundUp;
		});
		*/
	});
	~timeslider.setColors(~stringcolor,~stringcolor);


	/*~spacing1 = {(~firstviewboundsWIDTH2/72) + (~firstviewboundsWIDTH2/18 * ~line_multiple1)};
	~spacing2 = {(~firstviewboundsHEIGHT2/21*~line_multiple2)};

	if(~assist_part_number == nil, {~assist_part_number = 1});

	~new_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/3.7, ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["New Part",~stringcolor,~backgroundcolor]]).action_({arg button;

		~num_assist_parts = ~num_assist_parts + 1;
		if(~line_multiple2 >= 10, {~line_multiple1 = ~line_multiple1 + 1; ~line_multiple2 = 0;});
		~line_multiple2 = ~line_multiple2 + 1;

		("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirPut(
			Button.new(~firstview2, Rect(~spacing1.value, ~spacing2.value, ~firstviewboundsWIDTH/18, ~firstviewboundsHEIGHT/21)).states_([[~num_assist_parts.asString,~stringcolor,~backgroundcolor],[~num_assist_parts.asString,Color.black,Color.green]]).action_({arg button;
				("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 0;
				button.value = 1;
				~assist_part_number = button.string.asInteger;
				if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
				AppClock.sched(0.25, {~playbutton.valueAction = 1;});
			});
		);
		("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirGet.font = Font(size: ~firstviewboundsWIDTH/48);

	});
	~new_part_button.font = Font(size: ~firstviewboundsWIDTH/48);*/

	~spacing1 = {(~firstviewboundsWIDTH2/18 * ~line_multiple2)};
	~spacing2 = {(~firstviewboundsHEIGHT2/21 * ~line_multiple1)};

	if(~assist_part_number == nil, {~assist_part_number = 1});

	~new_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/3.7, ~firstviewboundsWIDTH/9, ~firstviewboundsHEIGHT/21)).states_([["New Part",~stringcolor,~backgroundcolor]]).action_({arg button;

		~num_assist_parts = ~num_assist_parts + 1;
		if(~line_multiple2 >= 16, {~line_multiple1 = ~line_multiple1 + 1; ~line_multiple2 = 0;});
		~line_multiple2 = ~line_multiple2 + 1;

		("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirPut(
			Button.new(~firstview2, Rect(~spacing1.value, ~spacing2.value, ~firstviewboundsWIDTH/18, ~firstviewboundsHEIGHT/21)).states_([[~num_assist_parts.asString,~stringcolor,~backgroundcolor],[~num_assist_parts.asString,Color.black,Color.green]]).action_({arg button;
				("assist_comp_part_"++(~assist_part_number)++"_button").asSymbol.envirGet.value = 0;
				button.value = 1;
				~assist_part_number = button.string.asInteger;
				if(~duplicate_on == 1, {
					if(~dup_1_part_num == nil, {~dup_1_part_num = ~assist_part_number}, {
						("assist_comp_part_"++(~dup_1_part_num)++"_button").asSymbol.envirGet.value = 1;
						~dup_2_part_num = ~assist_part_number;
					});
				});
				if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
				if(("assisted_inst_"++(1)++"_array_part_"++(~assist_part_number)).asSymbol.envirGet != nil, {AppClock.sched(0.25, {~playbutton.valueAction = 1;});});
			});
		);
		("assist_comp_part_"++(~num_assist_parts)++"_button").asSymbol.envirGet.font = Font(size: ~firstviewboundsWIDTH/48);

	});
	~new_part_button.font = Font(size: ~firstviewboundsWIDTH/48);

	if(~line_multiple1 == nil, {~line_multiple1 = 1});
	if(~line_multiple2 == nil, {~line_multiple2 = 0});

	if(~num_assist_parts == nil, {~num_assist_parts = 0}, {
		~num_assist_parts_backup = ~num_assist_parts;
		~line_multiple1_back_up = ~line_multiple1;
		~line_multiple2_back_up = ~line_multiple2;
		~num_assist_parts = 0;
		~line_multiple1 = 1;
		~line_multiple2 = 0;
		~num_assist_parts_backup.do({~new_part_button.valueAction = 0;});
		~num_assist_parts = ~num_assist_parts_backup;
		~line_multiple1 = ~line_multiple1_back_up;
		~line_multiple2 = ~line_multiple2_back_up;
	});

	/*~num_assist_parts = 0;
	~line_multiple1 = 0;
	~line_multiple2 = 0;
	~part_limit_counter = 0;*/

	~delete_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2 + (~firstviewboundsWIDTH/9), ~firstviewboundsHEIGHT/3.7, ~firstviewboundsWIDTH/8, ~firstviewboundsHEIGHT/21)).states_([["Delete Part",~stringcolor,~backgroundcolor]]).action_({arg button;
		~assist_compose_page_func.value;

	});
	~delete_part_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~undo_delete_part_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3 + (~firstviewboundsWIDTH/8)+ (~firstviewboundsWIDTH/9), ~firstviewboundsHEIGHT/3.7, ~firstviewboundsWIDTH/5.8, ~firstviewboundsHEIGHT/21)).states_([["Undo Delete Part",~stringcolor,~backgroundcolor]]).action_({arg button;
		~assist_compose_page_func.value;

	});
	~undo_delete_part_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~load_piece_to_main_page_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*4 + (~firstviewboundsWIDTH/8)+ (~firstviewboundsWIDTH/9) + (~firstviewboundsWIDTH/5.8), ~firstviewboundsHEIGHT/3.7, ~firstviewboundsWIDTH/3.8, ~firstviewboundsHEIGHT/21)).states_([["Load Piece To Main Page",~stringcolor,~backgroundcolor]]).action_({arg button;
		~assist_compose_page_func.value;

	});
	~load_piece_to_main_page_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~generate_new_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.97, ~firstviewboundsWIDTH/7, ~firstviewboundsHEIGHT/21)).states_([["Generate New",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(~duranduranbot_routine.isPlaying == true, {~stop_func.value;});
		~assisted_comp_part_func.value;
		AppClock.sched(0.25, {~playbutton.valueAction = 1;});
	});
	~generate_new_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~undo_generate_new_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2 + (~firstviewboundsWIDTH/7), ~firstviewboundsHEIGHT/2.97, ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21)).states_([["Undo Generate New",~stringcolor,~backgroundcolor]]).action_({arg button;

	});
	~undo_generate_new_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~save_generate_new_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3 + (~firstviewboundsWIDTH/7) + (~firstviewboundsWIDTH/5), ~firstviewboundsHEIGHT/2.97, ~firstviewboundsWIDTH/5, ~firstviewboundsHEIGHT/21)).states_([["Save Generate New",~stringcolor,~backgroundcolor]]).action_({arg button;

	});
	~save_generate_new_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~set_duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/2.5, ~firstviewboundsWIDTH/6.6, ~firstviewboundsHEIGHT/21)).states_([["Set Duplicate",~stringcolor,~backgroundcolor],["Set Duplicate",~backgroundcolor,Color.green]]).action_({arg button;

		if(~dup_1_part_num != nil, {("assist_comp_part_"++(~dup_1_part_num)++"_button").asSymbol.envirGet.value = 0;});
		if(~dup_2_part_num != nil, {("assist_comp_part_"++(~dup_2_part_num)++"_button").asSymbol.envirGet.value = 0;});
		if(button.value == 1, {~duplicate_on = 1}, {
			~duplicate_on = 0;
			~dup_1_part_num = nil;
		});
	});
	~set_duplicate_button.font = Font(size: ~firstviewboundsWIDTH/48);

	if(~duplicate_on == 1, {~set_duplicate_button.value = 1;});

	~duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*2 + (~firstviewboundsWIDTH/6.6), ~firstviewboundsHEIGHT/2.5, ~firstviewboundsWIDTH/6.6, ~firstviewboundsHEIGHT/21)).states_([["Duplicate",~stringcolor,~backgroundcolor]]).action_({arg button;

		if(((~duplicate_on == 1) and: (~dup_1_part_num != nil) and: (~dup_2_part_num != nil)), {
			~number_of_instruments.do(s = 0; {
				s=s+1;
				("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
					("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
				);
				("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
					("assisted_inst_"++(s)++"_array_part_"++(~dup_1_part_num)).asSymbol.envirGet;
				);
			});
			("assist_comp_part_"++(~dup_1_part_num)++"_button").asSymbol.envirGet.value = 0;
			("assist_comp_part_"++(~dup_2_part_num)++"_button").asSymbol.envirGet.value = 0;
			~dup_1_part_num = nil;
			~duplicate_on = 0;
			~set_duplicate_button.value = 0;
		}, {"set duplicate not on\n - and/or no 2 parts where selected".postln;});
	});
	~duplicate_button.font = Font(size: ~firstviewboundsWIDTH/48);

	~undo_duplicate_button = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72*3 + (~firstviewboundsWIDTH/6.6*2), ~firstviewboundsHEIGHT/2.5, ~firstviewboundsWIDTH/6.6, ~firstviewboundsHEIGHT/21)).states_([["Undo Duplicate",~stringcolor,~backgroundcolor]]).action_({arg button;

		~number_of_instruments.do(s = 0; {
			s=s+1;
			//uses reverse forumla found in "~reversenewrandomseedsbutton" on main page
			("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirPut(
				("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirGet;
			);
			("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)).asSymbol.envirPut(
				("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirGet;
			);
			("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_1").asSymbol.envirPut(
				("assisted_inst_"++(s)++"_array_part_"++(~dup_2_part_num)++"_dup_save_2").asSymbol.envirGet;
			);
			"undo duplicate performed".postln;
		});
	});
	~undo_duplicate_button.font = Font(size: ~firstviewboundsWIDTH/48);

	/*fillertext = StaticText(~firstview, Rect(0, ~firstviewboundsHEIGHT/0.999999, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	fillertext.background=~backgroundcolor;
	fillertext.stringColor=~stringcolor;
	fillertext.string = "";*/

	/*fillertext2 = StaticText(~firstview2, Rect(0, ~firstviewboundsHEIGHT/0.999999, ~firstviewboundsWIDTH/2.2, ~firstviewboundsHEIGHT/21));
	fillertext2.background=~backgroundcolor;
	fillertext2.stringColor=~stringcolor;
	fillertext2.string = "";*/

};