library(tidyverse)
library(ggplot2)

df <- read.csv("result/city_sample/overall_0_summary.csv")

df %>% 
  pivot_longer(cols = !c(step,time_slot,population), names_to = "type", values_to = "value") %>%
  filter(type != "susceptible") %>%
  ggplot(aes(x = step, y = value, group = type, colour = type)) +
  geom_point() +
  geom_line()
