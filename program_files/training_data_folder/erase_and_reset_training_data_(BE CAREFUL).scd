



//USE THESE TO RESET TRAINING DATA.

///*****THESE WILL COMPLETELY ERASE AND RESET ALL TRANING DATA*****


// BOOT DDBOT BEFORE USING



/*



****************************BE CAREFUL*******************************
Interval training data resets


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












































//Experimental

/*	~number_of_instruments.do(w = 0; v = 0; {
w=w+1;
(~interval_array.size+1).do(y = (-1); {
y=y+1;
(~interval_array.size+1).do(z = (-1); {
z=z+1;
(~time_array.size+1).do(a = (-1); {
a=a+1;
(~time_array.size+1).do(b = (-1); {
b=b+1;
(~time_array.size+1).do(c = (-1); {
("player_"++(w)++"__"++(y)++"_"++(z)++"__"++(a)++"_"++(b)++"_"++(c=c+1)).asSymbol.envirPut(Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;);
g = File(Document.current.dir++"/ddbot_weights/player_"++(w)++"__"++(y)++"_"++(z)++"__"++(a)++"_"++(b)++"_"++(c)++".scd", "w");
~interval_array.size.do(s = (-1); {g.write(("player_"++(w)++"__"++(y)++"_"++(z)++"__"++(a)++"_"++(b)++"_"++(c)).asSymbol.envirGet.at(s=s+1).asString++" ");});
g.close;
v=v+1; v.postln;
});
});
});
});
});
});*/