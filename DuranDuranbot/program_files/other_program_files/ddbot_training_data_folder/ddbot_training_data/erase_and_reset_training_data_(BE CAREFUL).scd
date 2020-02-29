
/*


//USE THESE TO RESET TRAINING DATA.

///*****THESE WILL COMPLETELY ERASE AND RESET ALL TRANING DATA*****







****************************BE CAREFUL*******************************
Interval training data resets


j = 1;
j.do(w = 0; v = 0; {
w=w+1;
(~interval_array.size+1).do(z = (-1); {
("player_"++(w)++"__"++(z=z+1)++"_weights").asSymbol.envirPut(Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;);
g = File(Document.current.dir++"/ddbot_training_data_folder/ddbot_training_data_current_"++(j.asString)++"_inst/ddbot_interval_weights/player_"++(w)++"_interval_weights/"++(z)++"_weights.scd", "w");
~interval_array.size.do(s = (-1); {g.write(("player_"++(w)++"__"++(z)++"_weights").asSymbol.envirGet.at(s=s+1).asString++" ");});
g.close;
v=v+1;
});
v.postln;
});

/////////////////////////////////////////////

j = 2;
j.do(w = 0; v = 0; {
w=w+1;
(~interval_array.size+1).do(c = (-1); {
c=c+1;
(~interval_array.size+1).do(z = (-1); {
("player_"++(w)++"__"++(c)++"__"++(z=z+1)++"_weights").asSymbol.envirPut(Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;);
g = File(Document.current.dir++"/ddbot_training_data_folder/ddbot_training_data_current_"++(j.asString)++"_inst/ddbot_interval_weights/player_"++(w)++"_interval_weights/"++(c)++"__"++(z)++"_weights.scd", "w");
~interval_array.size.do(s = (-1); {g.write(("player_"++(w)++"__"++(c)++"__"++(z)++"_weights").asSymbol.envirGet.at(s=s+1).asString++" ");});
g.close;
v=v+1;
});
});
v.postln;
});

///////////////////////////////////////////

j = 3;
j.do(w = 0; v = 0; {
w=w+1;
(~interval_array.size+1).do(c = (-1); {
c=c+1;
(~interval_array.size+1).do(y = (-1); {
y=y+1;
(~interval_array.size+1).do(z = (-1); {
("player_"++(w)++"__"++(c)++"__"++(y)++"_"++(z=z+1)++"_weights").asSymbol.envirPut(Array.fill(~interval_array.size, {1/~interval_array.size}).normalizeSum;);
g = File(Document.current.dir++"/ddbot_training_data_folder/ddbot_training_data_current_"++(j.asString)++"_inst/ddbot_interval_weights/player_"++(w)++"_interval_weights/"++(c)++"__"++(y)++"_"++(z)++"_weights.scd", "w");
~interval_array.size.do(s = (-1); {g.write(("player_"++(w)++"__"++(c)++"__"++(y)++"_"++(z)++"_weights").asSymbol.envirGet.at(s=s+1).asString++" ");});
g.close;
v=v+1;
});
});
});
v.postln;
});

*/








/*
****************************BE CAREFUL*******************************
Time training data resets


j = 1;
j.do(w = 0; v = 0; {
w=w+1;
(~time_array.size+1).do(z = (-1); {
("player_"++(w)++"__"++(z=z+1)++"_time_weights").asSymbol.envirPut(Array.fill(~dna_choice_array.size, {1/~dna_choice_array.size}).normalizeSum;);
g = File(Document.current.dir++"/ddbot_training_data_folder/ddbot_training_data_current_"++(j.asString)++"_inst/ddbot_time_weights/player_"++(w)++"_time_weights/"++(z)++"_time_weights.scd", "w");
~dna_choice_array.size.do(s = (-1); {g.write(("player_"++(w)++"__"++(z)++"_time_weights").asSymbol.envirGet.at(s=s+1).asString++" ");});
g.close;
v=v+1;
});
v.postln;
});

////////////////////////////////////

j = 2;
j.do(w = 0; v = 0; {
w=w+1;
(~time_array.size+1).do(x = (-1); {
x=x+1;
(~time_array.size+1).do(z = (-1); {
("player_"++(w)++"__"++(x)++"_"++(z=z+1)++"_time_weights").asSymbol.envirPut(Array.fill(~dna_choice_array.size, {1/~dna_choice_array.size}).normalizeSum;);
g = File(Document.current.dir++"/ddbot_training_data_folder/ddbot_training_data_current_"++(j.asString)++"_inst/ddbot_time_weights/player_"++(w)++"_time_weights/"++(x)++"_"++(z)++"_time_weights.scd", "w");
~dna_choice_array.size.do(s = (-1); {g.write(("player_"++(w)++"__"++(x)++"_"++(z)++"_time_weights").asSymbol.envirGet.at(s=s+1).asString++" ");});
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
(~time_array.size+1).do(x = (-1); {
x=x+1;
(~time_array.size+1).do(y = (-1); {
y=y+1;
(~time_array.size+1).do(z = (-1); {
("player_"++(w)++"__"++(x)++"_"++(y)++"_"++(z=z+1)++"_time_weights").asSymbol.envirPut(Array.fill(~dna_choice_array.size, {1/~dna_choice_array.size}).normalizeSum;);
g = File(Document.current.dir++"/ddbot_training_data_folder/ddbot_training_data_current_"++(j.asString)++"_inst/ddbot_time_weights/player_"++(w)++"_time_weights/"++(x)++"_"++(y)++"_"++(z)++"_time_weights.scd", "w");
~dna_choice_array.size.do(s = (-1); {g.write(("player_"++(w)++"__"++(x)++"_"++(y)++"_"++(z)++"_time_weights").asSymbol.envirGet.at(s=s+1).asString++" ");});
g.close;
v=v+1;
});
});
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