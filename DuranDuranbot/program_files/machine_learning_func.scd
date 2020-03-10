// MACHINE LEARNING FUNCTIONS

// - "past" note interval ml goes past "rate_selection_start" point

~net_change = 0.02;

// interval functions

~interv_older_notes_func = {

	c = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
	b = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-2));
	if(((b != 0) and: (b != nil) and: (c != 0) and: (c != nil)), {
		while({c == b}, {
			q=q-1;
			c = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
			b = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-2));
		});
		if(((b != 0) and: (b != nil)), {
			e = c - b;
			if(e.isNegative == true, {e = e.neg});
			while({e > 12}, {e = e - 12});
			("interval_diff"++(u=u+1)).asSymbol.envirPut(e.value;);
		}, {
			e = 12;
			("interval_diff"++(u=u+1)).asSymbol.envirPut(e.value;);
		});
	});

};

~interv_harmony_func = {

	if(~number_of_instruments > 1, {
		(~number_of_instruments-1).do(t = 1; {
			e = j - ("instrument_"++(~inst_array.wrapAt(~inst_num-(t=t-1)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1));
			if(e.isNegative == true, {e = e.neg});
			while({e > 12}, {e = e - 12});
			("interval_diff"++(u=u+1)).asSymbol.envirPut(e.value;);
		});
	});

};





// Machine learning function for musical intervals

~interval_machine_learning_func = {

	(

		~savenum = 0;

		~interv_inst_case = {
			case
			{~number_of_instruments == 1}{(~interval_diff1.asString)++"__"++(~interval_diff2.asString)}
			{~number_of_instruments == 2}{(~interval_diff1.asString)++"__"++(~interval_diff2.asString)}
			{~number_of_instruments == 3}{(~interval_diff1.asString)++"__"++(~interval_diff2.asString)++"_"++(~interval_diff3.asString)};
		};

		// Machine learning for musical intervals

		("\n").post;
		("intervals written -").post;
		("\n").post;

		~number_of_instruments.do(r = 0; {
			r=r+1;
			~inst_num = r;
			(~rate_selection_end-~rate_selection_start).do(x = 0; {
				x=x+1;
				j = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1));
				v = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+x);
				if(((v != j) and: (v != 0) and: (j != 0) and: (j != nil)), {

					u = 0;

					// find interval of "newer" notes

					e = v - j;
					if(e.isNegative == true, {e = e.neg});
					while({e > 12}, {e = e - 12});
					("interval_diff"++(u)).asSymbol.envirPut(e.value;);

					// find interval of "older" notes

					q = x;
					~interv_older_notes_func.value;
					if(((b != 0) and: (b != nil)), {
						(case
							{~number_of_instruments == 1}{1}
							{~number_of_instruments == 2}{0}
							{~number_of_instruments == 3}{0};).do({
							q=q-1;
							~interv_older_notes_func.value;
						});
					});

					// find interval between intruments

					~interv_harmony_func.value;

					// other stuff

					(~number_of_instruments+1).do(u = (-1); {
						u=u+1;
						("interval_diff"++(u)).asSymbol.envirPut(("interval_diff"++(u)).asSymbol.envirGet.round.asInteger;);
					});

					((~interval_diff0).asString++" - "++(~player_array.wrapAt(r-1))++"__"++(~interv_inst_case.value)).postln;

					// write new training data to disk

					h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/interval_weights/"++(~player_array.wrapAt(r-1))++"/"++(~interv_inst_case.value)++"_weights.scd", r).interpret.normalizeSum;
					/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
					~interval_diff_revert_1.postln;*/
					h.put(~interval_diff0-1, (h.at(~interval_diff0-1) + (~net_change*~net_change_mult)));
					h = h.normalizeSum;
					g = File((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/interval_weights/"++(~player_array.wrapAt(r-1))++"/"++(~interv_inst_case.value)++"_weights.scd", "w");
					g.write(h.asString);
					g.close;
				});
			});
		});

		~savenum_interval = ~savenum;
	);

};

// bit length (time) functions

~older_times_ml_func = {

	k = 1;
	c = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
	b = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-2));
	if(((b != 0) and: (b != nil) and: (c != 0) and: (c != nil)), {
		while({c == b}, {
			q=q-1;
			c = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
			b = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-2));
			k = k + 1;
		});
		("length"++(u=u+1)).asSymbol.envirPut(k);
	});

};

~older_times_harmony_ml_func = {

	if(~number_of_instruments > 1, {
		(~number_of_instruments-1).do(t = 0; {
			q = x-1;
			d = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
			f = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+q);
			while({f == d}, {
				q=q-1;
				d = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(q-1));
				f = ("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+q);
			});
			c = (~rate_selection_start+(q+1));
			k = 1;
			while({
				("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(c=c+1) ==
				("instrument_"++(~inst_array.wrapAt(~inst_num-(t)))++"_full_midi").asSymbol.envirGet.at(c-1)
			}, {k = k +1});
			k=k+1;
			("length"++(u=u+1)).asSymbol.envirPut(k);
			t=t-1;
		});
	});

};

// Machine learning function for note times

~time_machine_learning_func = {

	(
		~savenum = 0;

		~time_inst_case = {
			case
			{~number_of_instruments == 1}{(~lengthplace1.asString)++"_"++(~lengthplace2.asString)}
			{~number_of_instruments == 2}{(~lengthplace1.asString)++"_"++(~lengthplace2.asString)}
			{~number_of_instruments == 3}{(~lengthplace1.asString)++"_"++(~lengthplace2.asString)++"_"++(~lengthplace3.asString)};
		};

		("\n").post;
		("bit durations written -").post;
		("\n").post;

		~number_of_instruments.do(r = 0; {
			r=r+1;
			~inst_num = r;
			(~rate_selection_end-~rate_selection_start).do(x = 0; {
				x=x+1;
				j = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+(x-1));
				v = ("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(~rate_selection_start+x);
				if(v != j, {

					u = 0;

					// find new bit length

					k = 1; c = (~rate_selection_start+(x));
					while({
						("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(c=c+1) ==
						("instrument_"++(~inst_num)++"_full_midi").asSymbol.envirGet.at(c-1)
					}, {k = k +1});
					("length"++(u)).asSymbol.envirPut(k);

					// find older bit lengths

					q = x;
					~older_times_ml_func.value;
					if(((b != 0) and: (b != nil)), {
						(case
							{~number_of_instruments == 1}{1}
							{~number_of_instruments == 2}{0}
							{~number_of_instruments == 3}{0};).do({
							q=q-1;
							~older_times_ml_func.value;
						});
					});

					// find bit lengths of other notes

					~older_times_harmony_ml_func.value;

					// other stuff

					(~number_of_instruments+1).do(u = (-1); {
						u=u+1;
						("length"++(u)).asSymbol.envirPut(("length"++(u)).asSymbol.envirGet.round.asInteger;);
					});

					(~number_of_instruments+1).do(u = (-1); {
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

					(~number_of_instruments+1).do(u = (-1); {
						u=u+1;
						("lengthplace"++(u)).asSymbol.envirPut(("lengthplace"++(u)).asSymbol.envirGet.round.asInteger;);
					});

					// don't think ~lengthplace matters with players. doesn't work like that in inst_clock_func.

					((~lengthplace0).asString++" - "++(~player_array.wrapAt(r-1))++" "++(~time_inst_case.value)).postln;

					// write new training data to disk

					h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/time_weights/"++(~player_array.wrapAt(r-1))++"/"++(~time_inst_case.value)++"_time_weights.scd", r).interpret.normalizeSum;
					("lengthplace_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++"__"++(~time_inst_case.value), h]);
					h.put(~lengthplace0, (h.at(~lengthplace0) + (~net_change*~net_change_mult)));
					h = h.normalizeSum;
					g = File((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/time_weights/"++(~player_array.wrapAt(r-1))++"/"++(~time_inst_case.value)++"_time_weights.scd", "w");
					g.write(h.asString);
					g.close;

					~rate_revert = 0;
				});
			});
		});

		~savenum_time = ~savenum;
	);

};

// Main machine learning function

~machine_learning_func = {

	if(~duranduranbot_routine.isPlaying == true, {~playbutton.valueAction = 0;});

	~interval_machine_learning_func.value;
	~time_machine_learning_func.value;

	AppClock.sched(0.1, {
		if(~duranduranbot_routine.isPlaying == false, {~playbutton.valueAction = 1; "pause".postln;});

		if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});
		~ratedonetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.063829787234, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/4.04545454545, ~firstviewboundsHEIGHT/21));
		~ratedonetext.background=~backgroundcolor;
		~ratedonetext.stringColor_(~stringcolor);
		~ratedonetext.string = "PIECE SELECTION RATED";
		~ratedonetext.font = Font(size: ~firstviewboundsWIDTH/55);

		AppClock.sched(3, {if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});});
	});

};





// Experimental "revert" machine learning function - to undo a "rate".


/*
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
h = FileReader.read((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/interval_weights/"++(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0))++"_weights.scd", false, true);
h = h.flatten.asFloat;
("interval_diff_revert2_"++(r)).asSymbol.envirPut([(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0)), h]);
});
~savenum_time.do(r = 0; {
r = r+1;
h = FileReader.read((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/time_weights/"++(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0))++"_time_weights.scd", false, true);
h = h.flatten.asFloat;
("lengthplace_revert2_"++(r)).asSymbol.envirPut([(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0)), h]);
});

~savenum_interval.do(r = 0; {
r = r+1;
h = ("interval_diff_revert_"++(r)).asSymbol.envirGet.at(1);
g = File((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/interval_weights/"++(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0))++"_weights.scd", "w");
~interval_array.size.do(s = (-1); {g.write(h.at(s=s+1).asString++" ");});
g.close;
});
~savenum_time.do(r = 0; {
r = r+1;
h = ("lengthplace_revert_"++(r)).asSymbol.envirGet.at(1);
g = File((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/time_weights/"++(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0))++"_time_weights.scd", "w");
~bit_choice_array.size.do(s = (-1); {g.write(h.at(s=s+1).asString++" ");});
g.close;
});

~savenum_interval.do(r = 0; {
r = r+1;
h = FileReader.read((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/interval_weights/"++(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0))++"_weights.scd", false, true);
h = h.flatten.asFloat;
("interval_diff_revert_"++(r)).asSymbol.envirPut([(("interval_diff_revert_"++(r)).asSymbol.envirGet.at(0)), (("interval_diff_revert2_"++(r)).asSymbol.envirGet.at(1))]);
});
~savenum_time.do(r = 0; {
r = r+1;
h = FileReader.read((~ddbot_root)++"/program_files/training_data_folder/ddbot_training_data/training_data_current_"++(~number_of_instruments.asString)++"_inst/time_weights/"++(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0))++"_time_weights.scd", false, true);
h = h.flatten.asFloat;
("lengthplace_revert_"++(r)).asSymbol.envirPut([(("lengthplace_revert_"++(r)).asSymbol.envirGet.at(0)), (("lengthplace_revert2_"++(r)).asSymbol.envirGet.at(1))]);
});

if(~rate_revert == 0, {"Rates Reverted".postln; ~rate_revert = 1;}, {"Rates Reverted To Last Rated".postln; ~rate_revert = 0;});
*/
AppClock.sched(0.25, {
if(~duranduranbot_routine.isPlaying == true, {if(~pause_piece == 1, {~playbutton.valueAction = 1});});

if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});
~ratedonetext = StaticText(~firstview, Rect(~firstviewboundsWIDTH/3.063829787234, ~firstviewboundsHEIGHT/4.228571428571, ~firstviewboundsWIDTH/4.04545454545, ~firstviewboundsHEIGHT/21));
~ratedonetext.background=~stringcolor;
~ratedonetext.stringColor_(~backgroundcolor);
~ratedonetext.string = "RATES REVERTED";
~ratedonetext.font = Font(size: ~firstviewboundsWIDTH/55);

AppClock.sched(3, {if(~ratedonetext != nil, {~ratedonetext.close; ~ratedonetext = nil;});});
});
};
*/