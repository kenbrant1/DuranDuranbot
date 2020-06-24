~instrument_menu_array_func = {
	[
		"Alesis-Sanctuary-QCard-Choir-Aah",
		"Alesis-Sanctuary-QCard-Promenade",
		"Korg-M1-Angels",
		"Korg-M3R-Chorale",
		"Korg-M3R-VoiceChoir",
		"Korg-Poly-800-Organ",
		"Korg-Poly-800-samp_2",
		"Roland-JV-2080-Choir-Aahs",
		"Roland JX-3P samp 2",
		"Yamaha-SY22-Itopia",
		"Yamaha-TG500-CH-Aah",
		"Yamaha-TG500-CH-Spirit"

	];
};

~player_settings_page_func = {

	if(~firstview != nil, {~firstview.close;});
	if(~firstview2 != nil, {~firstview2.close;});

	~firstview = ScrollView(w, Rect(0, 0, w.bounds.width, w.bounds.height));
	/*view.clearOnRefresh = false;*/
	~firstview.background = ~backgroundcolor;
	~firstviewboundsWIDTH = ~firstview.bounds.width;
	~firstviewboundsHEIGHT = ~firstview.bounds.height;

	~firstviewtext1 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/2.2857142857143, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/3, ~firstviewboundsHEIGHT/21));
	~firstviewtext1.background=~backgroundcolor;
	~firstviewtext1.stringColor=~stringcolor;
	~firstviewtext1.string = "Settings (scroll down for more)";
	~firstviewtext1.align = \topLeft;
	~firstviewtext1.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext2 = Button.new(~firstview, Rect(~firstviewboundsWIDTH/72, ~firstviewboundsHEIGHT/43.7, ~firstviewboundsWIDTH/16, ~firstviewboundsHEIGHT/21)).states_([["back",~stringcolor,~backgroundcolor],["back",~stringcolor,~backgroundcolor]]).action_({arg button;
		~firstview.close;
		if(~ai_assist_compose != 1, {~main_page_func.value;}, {
			~firstview.close;
			~assist_compose_page_func.value;
		});
	});
	~firstviewtext2.string = "Back";
	~firstviewtext2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~button_base = {(~firstviewboundsHEIGHT/9+(((~firstviewboundsHEIGHT/3)-(~firstviewboundsHEIGHT/3.7))*x))};

	("synth_choice_menu_"++(1)).asSymbol.envirPut(PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, (x=0; ~button_base.value;), ~firstviewboundsWIDTH/2, ~firstviewboundsHEIGHT/21)));
	("synth_choice_menu_"++(1)).asSymbol.envirGet.items = ["Current - "++(("synth_choice_"++(1)).asSymbol.envirGet.asString)]++(~instrument_menu_array_func.value);
	("synth_choice_menu_"++(1)).asSymbol.envirGet.background_(~backgroundcolor);
	("synth_choice_menu_"++(1)).asSymbol.envirGet.stringColor_(~stringcolor);
	("synth_choice_menu_"++(1)).asSymbol.envirGet.value = ("synth_choice_"++(1)).asSymbol.envirGet.asString;
	("synth_choice_menu_"++(1)).asSymbol.envirGet.action = { arg menu;

		if(menu.value != 0, {

			~synth_inst_num = 1;
			("synth_choice_"++(1)).asSymbol.envirPut(("synth_choice_menu_"++(1)).asSymbol.envirGet.items.at(menu.value).asString;);
			g = File((~ddbot_root)++"/program_files/other_program_files/synth_choice_"++(1)++".scd", "w");
			g.write(("synth_choice_"++(1)).asSymbol.envirGet.asString.quote;);
			g.close;

			~synthdefs_func.value;
		});

	};
	("synth_choice_menu_"++(1)).asSymbol.envirGet.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext8 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72*2 +(~firstviewboundsWIDTH/2), (~button_base.value;), ~firstviewboundsWIDTH/1.4, ~firstviewboundsHEIGHT/9));
	~firstviewtext8.background=~backgroundcolor;
	~firstviewtext8.stringColor=~stringcolor;
	~firstviewtext8.string = "Choose instrument "++(1)++" synth";
	~firstviewtext8.align = \topLeft;
	~firstviewtext8.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	("synth_choice_menu_"++(2)).asSymbol.envirPut(PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, (x=1; ~button_base.value;), ~firstviewboundsWIDTH/2, ~firstviewboundsHEIGHT/21)));
	("synth_choice_menu_"++(2)).asSymbol.envirGet.items = ["Current - "++(("synth_choice_"++(2)).asSymbol.envirGet.asString)]++(~instrument_menu_array_func.value);
	("synth_choice_menu_"++(2)).asSymbol.envirGet.background_(~backgroundcolor);
	("synth_choice_menu_"++(2)).asSymbol.envirGet.stringColor_(~stringcolor);
	("synth_choice_menu_"++(2)).asSymbol.envirGet.value = ("synth_choice_"++(2)).asSymbol.envirGet.asString;
	("synth_choice_menu_"++(2)).asSymbol.envirGet.action = { arg menu;

		if(menu.value != 0, {

			~synth_inst_num = 2;
			("synth_choice_"++(2)).asSymbol.envirPut(("synth_choice_menu_"++(2)).asSymbol.envirGet.items.at(menu.value).asString;);
			g = File((~ddbot_root)++"/program_files/other_program_files/synth_choice_"++(2)++".scd", "w");
			g.write(("synth_choice_"++(2)).asSymbol.envirGet.asString.quote;);
			g.close;

			~synthdefs_func.value;
		});

	};
	("synth_choice_menu_"++(2)).asSymbol.envirGet.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext9 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72*2 +(~firstviewboundsWIDTH/2), (~button_base.value;), ~firstviewboundsWIDTH/1.4, ~firstviewboundsHEIGHT/9));
	~firstviewtext9.background=~backgroundcolor;
	~firstviewtext9.stringColor=~stringcolor;
	~firstviewtext9.string = "Choose instrument "++(2)++" synth";
	~firstviewtext9.align = \topLeft;
	~firstviewtext9.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	("synth_choice_menu_"++(3)).asSymbol.envirPut(PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, (x=2; ~button_base.value;), ~firstviewboundsWIDTH/2, ~firstviewboundsHEIGHT/21)));
	("synth_choice_menu_"++(3)).asSymbol.envirGet.items = ["Current - "++(("synth_choice_"++(3)).asSymbol.envirGet.asString)]++(~instrument_menu_array_func.value);
	("synth_choice_menu_"++(3)).asSymbol.envirGet.background_(~backgroundcolor);
	("synth_choice_menu_"++(3)).asSymbol.envirGet.stringColor_(~stringcolor);
	("synth_choice_menu_"++(3)).asSymbol.envirGet.value = ("synth_choice_"++(3)).asSymbol.envirGet.asString;
	("synth_choice_menu_"++(3)).asSymbol.envirGet.action = { arg menu;

		if(menu.value != 0, {

			~synth_inst_num = 3;
			("synth_choice_"++(3)).asSymbol.envirPut(("synth_choice_menu_"++(3)).asSymbol.envirGet.items.at(menu.value).asString;);
			g = File((~ddbot_root)++"/program_files/other_program_files/synth_choice_"++(3)++".scd", "w");
			g.write(("synth_choice_"++(3)).asSymbol.envirGet.asString.quote;);
			g.close;

			~synthdefs_func.value;
		});

	};
	("synth_choice_menu_"++(3)).asSymbol.envirGet.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext10 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72*2 +(~firstviewboundsWIDTH/2), (~button_base.value;), ~firstviewboundsWIDTH/1.4, ~firstviewboundsHEIGHT/9));
	~firstviewtext10.background=~backgroundcolor;
	~firstviewtext10.stringColor=~stringcolor;
	~firstviewtext10.string = "Choose instrument "++(3)++" synth";
	~firstviewtext10.align = \topLeft;
	~firstviewtext10.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~comp_setting_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, (x=3; ~button_base.value;), ~firstviewboundsWIDTH/1.1, ~firstviewboundsHEIGHT/21));
	~comp_setting_menu.items = [
		"Current - "++(
			case
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/composition_process_setting.scd", false, true).flatten.at(0) == 1.asString}{"Use ML data and composition algorithm for composing (recommended)"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/composition_process_setting.scd", false, true).flatten.at(0) == 2.asString}{"Use just ML data for composing"};
		),
		"Use ML data and composition algorithm for composing (recommended)", "Use just ML data for composing"];
	~comp_setting_menu.background_(~backgroundcolor);
	~comp_setting_menu.stringColor_(~stringcolor);
	~comp_setting_menu.value = FileReader.read((~ddbot_root)++"/program_files/other_program_files/composition_process_setting.scd", false, true);
	~comp_setting_menu.action = { arg menu;

		case
		{~comp_setting_menu.value == 0}{nil}
		{~comp_setting_menu.value == 1}{
			~composition_process_setting = 1;
			g = File((~ddbot_root)++"/program_files/other_program_files/composition_process_setting.scd", "w");
			g.write(~composition_process_setting.asString);
			g.close;
		}
		{~comp_setting_menu.value == 2}{
			~composition_process_setting = 2;
			g = File((~ddbot_root)++"/program_files/other_program_files/composition_process_setting.scd", "w");
			g.write(~composition_process_setting.asString);
			g.close;
		};
	};
	~comp_setting_menu.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~tuningfreqtextfield = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, (x=4; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~tuningfreqtextfield.string = ~tuning_frequency.asString;
	~tuningfreqtextfield.background_(~backgroundcolor);
	~tuningfreqtextfield.stringColor_(~stringcolor);
	~tuningfreqtextfield.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~tuningfreqtextfield.action = {arg value;
		if(~tuningfreqtextfield.value.interpret.isNumber == true, {
			~tuning_frequency = ~tuningfreqtextfield.value.asFloat;
			g = File((~ddbot_root)++"/program_files/other_program_files/tuning_frequency.scd", "w");
			g.write(~tuning_frequency.asString);
			g.close;
			~tuningfreqtextfield2.string = "Note A4 tuning frquency in hertz. Current - "++(~tuning_frequency)++". Original default is "++(~tuning_frequency_default)++".";
		}, {
			~tuningfreqtextfield2.string = "Please use a number.";
			AppClock.sched(4, {~tuningfreqtextfield2.string = "Note A4 tuning frquency in hertz. Current - "++(~tuning_frequency)++". Original default is "++(~tuning_frequency_default)++".";});
		});
	};

	~tuningfreqtextfield2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/8, (x=4; ~button_base.value;), ~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/21));
	~tuningfreqtextfield2.background=~backgroundcolor;
	~tuningfreqtextfield2.stringColor=~stringcolor;
	~tuningfreqtextfield2.string = "Note A4 tuning frquency in hertz. Current - "++(~tuning_frequency)++". Original default is "++(~tuning_frequency_default)++".";
	~tuningfreqtextfield2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~playerspeedtextfield = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, (x=5; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
	~playerspeedtextfield.string = ~bit_duration.reciprocal.asString;
	~playerspeedtextfield.background_(~backgroundcolor);
	~playerspeedtextfield.stringColor_(~stringcolor);
	~playerspeedtextfield.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	~playerspeedtextfield.action = {
		/*randseed_1_make = randseedEZText1.value;
		randseed_1_make = randseed_1_make.asString++"";
		randseed_1_make = randseed_1_make.separate;*/
		/*if(i.size > 8, {num = nil; playerspeedtextfield.string = "must be 8 digits or less";}, {*/
		~bit_duration = ~playerspeedtextfield.value.asFloat.reciprocal;
		g = File((~ddbot_root)++"/program_files/other_program_files/bit_speed.scd", "w");
		g.write(~bit_duration.asString);
		g.close;
		~playerspeedtextfield2.string = "Bit speed of piece. Input how fast "++"bits".quote++" are cycled through in seconds. Then press "++"Enter/Return".quote++". Current - "++(~bit_duration.reciprocal)++". Original default is "++(~bit_duration_default)++".";

	};

	~playerspeedtextfield2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/8,(x=5; ~button_base.value;), ~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/11));
	~playerspeedtextfield2.background=~backgroundcolor;
	~playerspeedtextfield2.stringColor=~stringcolor;
	~playerspeedtextfield2.string = "Bit speed of piece. Input how fast "++"bits".quote++" are cycled through in seconds. Then press "++"Enter/Return".quote++". Current - "++(~bit_duration.reciprocal)++". Original default is "++(~bit_duration_default)++".";
	~playerspeedtextfield2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~num_instruments_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
	~num_instruments_menu.items = [
		"Current - "++(
			case
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 1.asString}{"1 voice"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 2.asString}{"2 voices"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true).flatten.at(0) == 3.asString}{"3 voices"};
		),
		"1 voice", "2 voices", "3 voices"];
	~num_instruments_menu.background_(~backgroundcolor);
	~num_instruments_menu.stringColor_(~stringcolor);
	~num_instruments_menu.value = FileReader.read((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", false, true);
	~num_instruments_menu.action = { arg menu;

		if(~duranduranbot_routine.isPlaying == true, {
			~stopbutton.close;
			~playbutton.value = 0;
			~duranduranbot_routine.stop;
			Server.default.freeAll;
		});
		w.close;

		case
		{~num_instruments_menu.value == 0}{nil}
		{~num_instruments_menu.value == 1}{
			~number_of_instruments = 1;
			g = File((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", "w");
			g.write(~number_of_instruments.asString);
			g.close;
		}
		{~num_instruments_menu.value == 2}{
			~number_of_instruments = 2;
			g = File((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", "w");
			g.write(~number_of_instruments.asString);
			g.close;
		}
		{~num_instruments_menu.value == 3}{
			~number_of_instruments = 3;
			g = File((~ddbot_root)++"/program_files/other_program_files/numberofvoices.scd", "w");
			g.write(~number_of_instruments.asString);
			g.close;
		};

		~duranduranbot_GUI_func.value;

	};
	~num_instruments_menu.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext3 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.6, (x=7; ~button_base.value;), ~firstviewboundsWIDTH/1.4, ~firstviewboundsHEIGHT/9));
	~firstviewtext3.background=~backgroundcolor;
	~firstviewtext3.stringColor=~stringcolor;
	~firstviewtext3.string = "Set the default setting of how many voices/intruments are playing. Original default is 2.";
	~firstviewtext3.align = \topLeft;
	~firstviewtext3.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~firstviewtext4 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/72, (x=9; ~button_base.value;), ~firstviewboundsWIDTH/1.0285714285714, ~firstviewboundsHEIGHT/6.5));
	~firstviewtext4.background=~backgroundcolor;
	~firstviewtext4.stringColor=~stringcolor;
	~firstviewtext4.string = "Click on the pop-up menu below to choose which player(s) will compose and be trained. This is useful for a more targeted composition and training... or less targeted. By selecting an option from the pop-up menu below, you are making that the default setting. Original default is "++"Player 1".quote++".";
	~firstviewtext4.align = \topLeft;
	~firstviewtext4.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~players_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, (x=12; ~button_base.value;), ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
	~players_menu.items = [
		"Current - "++(
			case
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 0.asString}{"All Players"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 1.asString}{"Player 1"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 2.asString}{"Player 2"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true).flatten.at(0) == 3.asString}{"Player 3"};
		),
		"All Players", "Player 1", "Player 2", "Player 3"];
	~players_menu.background_(~backgroundcolor);
	~players_menu.stringColor_(~stringcolor);
	~players_menu.value = FileReader.read((~ddbot_root)++"/program_files/other_program_files/which_players.scd", false, true);
	~players_menu.action = { arg menu;

		case
		{~players_menu.value == 0}{nil}
		{~players_menu.value == 1}{~which_players = "0".asString}
		{~players_menu.value == 2}{~which_players = "1".asString}
		{~players_menu.value == 3}{~which_players = "2".asString}
		{~players_menu.value == 4}{~which_players = "3".asString};

		g = File((~ddbot_root)++"/program_files/other_program_files/which_players.scd", "w");
		g.write(~which_players.asString;);
		g.close;
	};
	~players_menu.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	~color_menu = PopUpMenu(~firstview, Rect(~firstviewboundsWIDTH/72, (x=13; ~button_base.value;), ~firstviewboundsWIDTH/4, ~firstviewboundsHEIGHT/21));
	~color_menu.items = [
		"GUI color - "++(
			case
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true).flatten.at(0) == 0.asString}{"Black"}
			{FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true).flatten.at(0) == 1.asString}{"White"};
		),
		"Black", "White"];
	~color_menu.background_(~backgroundcolor);
	~color_menu.stringColor_(~stringcolor);
	~color_menu.value = FileReader.read((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", false, true);
	~color_menu.action = { arg menu;
		g = File((~ddbot_root)++"/program_files/other_program_files/gui_color.scd", "w");
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
	~color_menu.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);

	if(~ai_assist_compose != 1, {

		~rate_time_range_textfield = TextField(~firstview, Rect(~firstviewboundsWIDTH/72, (x=14; ~button_base.value;), ~firstviewboundsWIDTH/10, ~firstviewboundsHEIGHT/21));
		~rate_time_range_textfield.string = ~rate_time_range.asString;
		~rate_time_range_textfield.background_(~backgroundcolor);
		~rate_time_range_textfield.stringColor_(~stringcolor);
		~rate_time_range_textfield.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
		~rate_time_range_textfield.action = {
			/*randseed_1_make = randseedEZText1.value;
			randseed_1_make = randseed_1_make.asString++"";
			randseed_1_make = randseed_1_make.separate;*/
			/*if(i.size > 8, {num = nil; playerspeedtextfield.string = "must be 8 digits or less";}, {*/
			~rate_time_range = ~rate_time_range_textfield.value.asFloat;
			g = File((~ddbot_root)++"/program_files/other_program_files/rate_time_range.scd", "w");
			g.write(~rate_time_range.asString);
			g.close;
			~rate_time_range_textfield2.string = "Rate time range. Input time range in seconds that is rated backwards from current piece position. Then press "++"Enter/Return".quote++". Current - "++(~rate_time_range)++". Original default is "++(~rate_time_range_default)++".";

		};

		~rate_time_range_textfield2 = StaticText(~firstview, Rect(~firstviewboundsWIDTH/8, (x=14; ~button_base.value;), ~firstviewboundsWIDTH/1.2, ~firstviewboundsHEIGHT/11));
		~rate_time_range_textfield2.background=~backgroundcolor;
		~rate_time_range_textfield2.stringColor=~stringcolor;
		~rate_time_range_textfield2.string = "Rate time range. Input time range in seconds that is rated backwards from current piece position. Then press "++"Enter/Return".quote++". Current - "++(~rate_time_range)++". Original default is "++(~rate_time_range_default)++".";
		~rate_time_range_textfield2.font = Font(~default_font, size: ~firstviewboundsWIDTH/48);
	});
};