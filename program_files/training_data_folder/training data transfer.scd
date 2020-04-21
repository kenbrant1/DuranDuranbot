
12.do(s=(-1); {
	s=s+1;
	12.do(d=(-1); {
		d=d+1;
h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/compose_ml_data/"++(~interval_diff1)++"__"++(~interval_diff2)++".scd", r).interpret.normalizeSum;
/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
~interval_diff_revert_1.postln;*/
h.put(~interval_diff0+(~blend_choice_array.at(0).neg), (h.at(~interval_diff0+(~blend_choice_array.at(0).neg)) + (~net_change*~net_change_mult)));
h = h.normalizeSum;
g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/compose_ml_data/"++(~interval_diff1)++"__"++(~interval_diff2)++".scd", "w");
g.write(h.asString);
g.close;
	});
});


~array = [];
12.do(s=(-1); {
	s=s+1;
	12.do(d=(-1); {
		d=d+1;
		h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/compose_ml_data/"++(s)++"__"++(d)++".scd", r).interpret.normalizeSum;
		/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
		~interval_diff_revert_1.postln;*/
		/*h.put(~interval_diff0+(~blend_choice_array.at(0).neg), (h.at(~interval_diff0+(~blend_choice_array.at(0).neg)) + (~net_change*~net_change_mult)));
		h = h.normalizeSum;*/
		~array = ~array.add(h);
	});
});
g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/new/compose_ml_data.scd", "w");
g.write("[\n");
12.do(s=(-1); f = (-1); {
	s=s+1;
	12.do(d=(-1); {
		d=d+1;
		f=f+1;
		g.write(("/*"++(s)++"__"++(d)++"*/ "++(~array.at(f))).asString++",\n");
	});
});
g.write("]");
g.close;






~array = [];
13.do(s=(-1); {
	s=s+1;
	13.do(d=(-1); {
		d=d+1;
		h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_3_inst/interval_weights/player_1/"++(s)++"__"++(d)++"_weights.scd", r).interpret.normalizeSum;
		/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
		~interval_diff_revert_1.postln;*/
		/*h.put(~interval_diff0-1, (h.at(~interval_diff0-1) + (~net_change*~net_change_mult)));
		h = h.normalizeSum;*/
		~array = ~array.add(h);
	});
});
g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/new/3_inst_player_1_interval_weights.scd", "w");
g.write("[
");
13.do(s=(-1); f = (-1); {
	s=s+1;
	13.do(d=(-1); {
		d=d+1;
		f=f+1;
		g.write(("/*"++(s)++"__"++(d)++"*/ "++(~array.at(f))).asString++",\n");
	});
});
g.write("]");
g.close;

~array = [];
13.do(s=(-1); {
	s=s+1;
	13.do(d=(-1); {
		d=d+1;
		13.do(j=(-1); {
			j=j+1;
			h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_3_inst/interval_weights/player_1/"++(s)++"__"++(d)++"_"++(j)++"_weights.scd", r).interpret.normalizeSum;
			/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
			~interval_diff_revert_1.postln;*/
			/*h.put(~interval_diff0-1, (h.at(~interval_diff0-1) + (~net_change*~net_change_mult)));
			h = h.normalizeSum;*/
			~array = ~array.add(h);
		});
	});
});
g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/new/3_inst_player_1_interval_weights.scd", "w");
g.write("[
");
13.do(s=(-1); f = (-1); {
	s=s+1;
	13.do(d=(-1); {
		d=d+1;
		13.do(j=(-1); {
			j=j+1;
			f=f+1;
			g.write(("/*"++(s)++"__"++(d)++"_"++(j)++"*/ "++(~array.at(f))).asString++",
");
		});
	});
});
g.write("]");
g.close;



~array = [];
13.do(s=(-1); {
	s=s+1;
	13.do(d=(-1); {
		d=d+1;
		h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_1_inst/interval_weights/player_1/"++(s)++"__"++(d)++"_weights.scd", r).interpret.normalizeSum;
		/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
		~interval_diff_revert_1.postln;*/
		/*h.put(~interval_diff0-1, (h.at(~interval_diff0-1) + (~net_change*~net_change_mult)));
		h = h.normalizeSum;*/
		~array = ~array.add(h);
	});
});
g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/new/1_inst_player_1_interval_weights.scd", "w");
13.do(s=(-1); f = (-1); {
	s=s+1;
	13.do(d=(-1); {
		d=d+1;
		f=f+1;
		g.write(("/*"++(s)++"__"++(d)++"*/ "++(~array.at(f))).asString++"
");
	});
});
g.close;





~array = [];
9.do(s=(-1); {
	s=s+1;
	9.do(d=(-1); {
		d=d+1;
		h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_3_inst/time_weights/player_1/"++(s)++"_"++(d)++"_time_weights.scd", r).interpret.normalizeSum;
		/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
		~interval_diff_revert_1.postln;*/
		/*h.put(~interval_diff0-1, (h.at(~interval_diff0-1) + (~net_change*~net_change_mult)));
		h = h.normalizeSum;*/
		~array = ~array.add(h);
	});
});
g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/new/3_inst_player_1_time_weights.scd", "w");
g.write("[\n");
9.do(s=(-1); f = (-1); {
	s=s+1;
	9.do(d=(-1); {
		d=d+1;
		f=f+1;
		g.write(("/*"++(s)++"_"++(d)++"*/ "++(~array.at(f))).asString++",\n");
	});
});
g.write("]");
g.close;



~array = [];
9.do(s=(-1); {
	s=s+1;
	9.do(d=(-1); {
		d=d+1;
		9.do(j=(-1); {
			j=j+1;
			h = File.readAllString((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_3_inst/time_weights/player_1/"++(s)++"_"++(d)++"_"++(j)++"_time_weights.scd", r).interpret.normalizeSum;
			/*("interval_diff_revert_"++(~savenum=~savenum+1)).asSymbol.envirPut([(~player_array.wrapAt(r-1))++" "++(inst_case.value)), h]);
			~interval_diff_revert_1.postln;*/
			/*h.put(~interval_diff0-1, (h.at(~interval_diff0-1) + (~net_change*~net_change_mult)));
			h = h.normalizeSum;*/
			~array = ~array.add(h);
		});
	});
});
g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/new/3_inst_player_1_time_weights.scd", "w");
g.write("[\n");
9.do(s=(-1); f = (-1); {
	s=s+1;
	9.do(d=(-1); {
		d=d+1;
		9.do(j=(-1); {
			j=j+1;
			f=f+1;
			g.write(("/*"++(s)++"_"++(d)++"_"++(j)++"*/ "++(~array.at(f))).asString++",\n");

		});
	});
});
g.write("]");
g.close;






















j = 1;
j.do(w = 0; v = 0; {
	w=w+1;
	(~interval_array.size+1).do(c = (-1); {
		c=c+1;
		(~interval_array.size+1).do(z = (-1); {
			z=z+1;
			f = Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;
			g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_"++(j.asString)++"_inst/interval_weights/player_"++(w)++"/"++(c)++"__"++(z)++"_weights.scd", "w");
			g.write(f.asString);
			g.close;
			v=v+1;
		});
	});
	v.postln;
});


j = 2;
j.do(w = 0; v = 0; {
	w=w+1;
	(~interval_array.size+1).do(c = (-1); {
		c=c+1;
		(~interval_array.size+1).do(z = (-1); {
			z=z+1;
			f = Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;
			g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_"++(j.asString)++"_inst/interval_weights/player_"++(w)++"/"++(c)++"__"++(z)++"_weights.scd", "w");
			g.write(f.asString);
			g.close;
			v=v+1;
		});
	});
	v.postln;
});


j = 3;
j.do(w = 0; v = 0; {
	w=w+1;
	(~interval_array.size+1).do(c = (-1); {
		c=c+1;
		(~interval_array.size+1).do(y = (-1); {
			y=y+1;
			(~interval_array.size+1).do(z = (-1); {
				z=z+1;
				f = Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;
				g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_"++(j.asString)++"_inst/interval_weights/player_"++(w)++"/"++(c)++"__"++(y)++"_"++(z)++"_weights.scd", "w");
				g.write(f.asString);
				g.close;
				v=v+1;
			});
		});
	});
	v.postln;
});










****************************BE CAREFUL*******************************
Time training data resets


j = 1;
j.do(w = 0; v = 0; {
	w=w+1;
	(~time_array.size+1).do(c = (-1); {
		c=c+1;
		(~time_array.size+1).do(z = (-1); {
			z=z+1;
			f = Array.fill(~bit_choice_array.size, {1/~bit_choice_array.size}).normalizeSum;
			g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_"++(j.asString)++"_inst/time_weights/player_"++(w)++"/"++(c)++"_"++(z)++"_time_weights.scd", "w");
			g.write(f.asString);
			g.close;
			v=v+1;
		});
	});
	v.postln;
});




j = 2;
j.do(w = 0; v = 0; {
	w=w+1;
	(~time_array.size+1).do(c = (-1); {
		c=c+1;
		(~time_array.size+1).do(z = (-1); {
			z=z+1;
			f = Array.fill(~bit_choice_array.size, {1/~bit_choice_array.size}).normalizeSum;
			g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_"++(j.asString)++"_inst/time_weights/player_"++(w)++"/"++(c)++"_"++(z)++"_time_weights.scd", "w");
			g.write(f.asString);
			g.close;
			v=v+1;
		});
	});
	v.postln;
});

////////////////////////////////////////

j = 3;
j.do(w = 0; v = 0; {
	w=w+1;
	(~time_array.size+1).do(c = (-1); {
		c=c+1;
		(~time_array.size+1).do(y = (-1); {
			y=y+1;
			(~time_array.size+1).do(z = (-1); {
			z=z+1;
			f = Array.fill(~bit_choice_array.size, {1/~bit_choice_array.size}).normalizeSum;
			g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/training_data_current_"++(j.asString)++"_inst/time_weights/player_"++(w)++"/"++(c)++"_"++(y)++"_"++(z)++"_time_weights.scd", "w");
			g.write(f.asString);
			g.close;
			v=v+1;
		});
		});
	});
	v.postln;
});











****************************BE CAREFUL*******************************
Compositon process interval training data resets

v=0;
(~comp_process_interval_array.size).do(c = (-1); {
	c=c+1;
	(~comp_process_interval_array.size).do(z = (-1); {
		z=z+1;
		f = Array.fill(~comp_process_interval_array.size+1, {1/~comp_process_interval_array.size}).normalizeSum;
		g = File((~ddbot_root)++"/program_files/training_data_folder/"++(~ddbot_training_data_location)++"/compose_ml_data/"++(c)++"__"++(z)++".scd", "w");
		g.write(f.asString);
		g.close;
		v=v+1;
	});
	v.postln;
});


*/


