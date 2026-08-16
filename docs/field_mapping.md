# Mapování polí: Czech → English → DB sloupec

Tento dokument popisuje mapování názvů polí z českých faktur na anglické názvy v kódu
a na sloupcové názvy v databázi.

---

## Společná pole (všechny komodity)

| České jméno | Popis | Python pole (entities.py) | DB sloupec (tabulka `invoices`) |
|---|---|---|---|
| doklad_cislo | Číslo faktury | `invoice_number` | `invoice_number` |
| kod_odberne_misto / EAN / EIC | Kód odběrného místa | `supply_point.ean_code` / `.eic_code` / `.consumption_point_code` | `supply_point_code` |
| obdobi_od | Fakturační období od | `period.period_from` | `period_from` |
| obdobi_do | Fakturační období do | `period.period_to` | `period_to` |
| datum_vystaveni | Datum vystavení | `issue_date` | `issue_date` |
| datum_splatnosti | Datum splatnosti | `due_date` | `due_date` |
| datum_uzp | Datum uskutečnění zdanitelného plnění (DUZP) | `vat_date` | `tax_point_date` |
| ico_odberatel | IČO odběratele | `customer_tax_id` | `customer_cin` |
| ico_dodavatele | IČO dodavatele | `supplier_tax_id` | `supplier_cin` |
| castka_bez_dph | Částka bez DPH (Kč) | `total_amount_ex_vat` | `total_amount_ex_vat` |
| castka_s_dph | Částka s DPH (Kč) | `total_amount_inc_vat` | `total_amount_inc_vat` |
| dan_zakladni | Výše DPH (Kč) | `vat_amount` | `vat_amount` |
| sazba_dph | Sazba DPH (%) | `vat_rate` | `vat_rate` |
| opravna | Opravná faktura | `is_correction` | `is_correction` |
| prechodova_faktura | Přechodová faktura | `is_transitional` | `is_transitional` |

---

## Elektřina NN (elektrina_nn)

DB tabulka: `electricity_nn_details`

| České jméno | Popis | Python pole (ElectricityNNData) | DB sloupec |
|---|---|---|---|
| spotreba_nt | Spotřeba nízkého tarifu (kWh) | `consumption_low_tariff` | `consumption_low_tariff` |
| spotreba_vt | Spotřeba vysokého tarifu (kWh) | `consumption_high_tariff` | `consumption_high_tariff` |
| celkova_spotreba | Celková spotřeba (kWh) | `total_consumption` | `total_consumption` |
| distribucni_tarif | Distribuční tarif (D01d, D02d, ...) | `distribution_tariff` | `distribution_tariff` |
| jistic | Jistič (A) | `circuit_breaker_value` | `circuit_breaker_value` |
| silova_elektrina | Silová elektřina (Kč) | `supply_charge` | `supply_charge` |
| distribuce | Distribuce (Kč) | `distribution_charge` | `distribution_charge` |
| systemove_sluzby | Systémové služby (Kč) | `system_services` | `system_services` |
| poze | POZE — podpora obnovitelných zdrojů energie (Kč) | `renewable_energy_fee` | `renewable_energy_fee` |
| castka_bez_dph | Částka bez DPH (Kč) | `amount_ex_vat` | `amount_ex_vat` |
| castka_s_dph | Částka s DPH (Kč) | `amount_inc_vat` | `amount_inc_vat` |

---

## Elektřina VN (elektrina_vn)

DB tabulka: `electricity_vn_details`

| České jméno | Popis | Python pole (ElectricityVNData) | DB sloupec |
|---|---|---|---|
| spotreba_se | Spotřeba SE (MWh) | `supply_consumption` | `supply_consumption` |
| castka_se | Částka SE (Kč) | `supply_charge` | `supply_charge` |
| castka_dan_se | Částka daň SE (Kč) | `supply_tax_charge` | `supply_tax_charge` |
| ctvrt_hod_max | Čtvrthodinové maximum (MW) | `quarter_hour_max` | `quarter_hour_max` |
| sazba_eru | Sazba ERÚ | `eru_rate` | `eru_rate` |
| rk_rocni | Roční rezervovaná kapacita (MW) | `annual_reserved_capacity` | `annual_reserved_capacity` |
| castka_rk_rocni | Částka RK roční (Kč) | `annual_reserved_capacity_charge` | `annual_reserved_capacity_charge` |
| rk_mesicni | Měsíční rezervovaná kapacita (MW) | `monthly_reserved_capacity` | `monthly_reserved_capacity` |
| castka_rk_mesicni | Částka RK měsíční (Kč) | `monthly_reserved_capacity_charge` | `monthly_reserved_capacity_charge` |
| sazba_pouziti_siti | Sazba použití sítí (Kč/MWh) | `grid_usage_rate` | `grid_usage_rate` |
| castka_pouziti_siti | Částka použití sítí (Kč) | `grid_usage_charge` | `grid_usage_charge` |
| prekroceni_rk | Překročení rezervované kapacity (MW) | `reserved_capacity_excess` | `reserved_capacity_excess` |
| sazba_prekroceni_rk | Sazba překročení RK (Kč/MW) | `reserved_capacity_excess_rate` | `reserved_capacity_excess_rate` |
| castka_prekroceni_rk | Částka překročení RK (Kč) | `reserved_capacity_excess_charge` | `reserved_capacity_excess_charge` |
| tg_fi_vn | Účiník tg φ | `power_factor` | `power_factor` |
| mnozstvi_jalova | Množství jalové energie (kVArh) | `reactive_power_quantity` | `reactive_power_quantity` |
| sazba_jalova | Sazba jalová (Kč/kVArh) | `reactive_power_rate` | `reactive_power_rate` |
| castka_jalova | Částka jalová (Kč) | `reactive_power_charge` | `reactive_power_charge` |
| cena_sluzby | Cena služby (Kč) | `service_price` | `service_price` |
| poze | POZE (Kč) | `renewable_energy_fee` | `renewable_energy_fee` |
| cena_provoz | Cena provozu (Kč) | `operating_price` | `operating_price` |
| castka_bez_dph | Částka bez DPH (Kč) | `amount_ex_vat` | `amount_ex_vat` |
| castka_s_dph | Částka s DPH (Kč) | `amount_inc_vat` | `amount_inc_vat` |

---

## Plyn MO — maloodběr (plyn_MO)

DB tabulka: `gas_mo_details`

| České jméno | Popis | Python pole (GasMOData) | DB sloupec |
|---|---|---|---|
| spotreba_m3 | Spotřeba (m³) | `consumption_m3` | `consumption_m3` |
| spotreba_mwh | Spotřeba (MWh) | `consumption_mwh` | `consumption_mwh` |
| koef_prepoctu | Přepočtový koeficient | `conversion_factor` | `conversion_factor` |
| spalne_teplo | Spalné teplo (MJ/m³) | `combustion_heat` | `combustion_heat` |
| obdobi_mesice | Počet měsíců v období | `period_months` | `period_months` |
| jedn_cena_komoditni_slozky | Jedn. cena komoditní složky (Kč/MWh) | `commodity_unit_price` | `commodity_unit_price` |
| cena_za_komoditni_slozku_ceny | Cena za komoditní složku (Kč) | `commodity_total_price` | `commodity_total_price` |
| jedn_cena_staly_mesicni_plat | Jedn. cena stálý měsíční plat (Kč/měs) | `fixed_monthly_fee_unit_price` | `fixed_monthly_fee_unit_price` |
| cena_za_staly_mesicni_plat | Stálý měsíční plat (Kč) | `fixed_monthly_fee` | `fixed_monthly_fee` |
| jedn_cena_distr_plynu | Jedn. cena distribuce plynu (Kč/MWh) | `distribution_unit_price` | `distribution_unit_price` |
| pevna_cena_za_distribuci_plynu | Pevná cena distribuce (Kč) | `distribution_fixed_price` | `distribution_fixed_price` |
| jedn_cena_pristavena_kapacita | Jedn. cena přistavená kapacita (Kč/m³/h) | `reserved_capacity_unit_price` | `reserved_capacity_unit_price` |
| cena_za_pristavenou_kapacitu | Cena za přistavenou kapacitu (Kč) | `reserved_capacity_price` | `reserved_capacity_price` |
| cena_za_cinnost_operatora_trhu | Cena za činnost operátora trhu (Kč) | `market_operator_price` | `market_operator_price` |
| dan_zemni_plyn_celkem | Daň ze zemního plynu celkem (Kč) | `natural_gas_tax_total` | `natural_gas_tax_total` |
| castka_bez_dph | Částka bez DPH (Kč) | `amount_ex_vat` | `amount_ex_vat` |
| castka_s_dph | Částka s DPH (Kč) | `amount_inc_vat` | `amount_inc_vat` |

---

## Plyn VO — velkoodběr (plyn_VO)

DB tabulka: `gas_vo_details`

| České jméno | Popis | Python pole (GasVOData) | DB sloupec |
|---|---|---|---|
| spotreba_m | Spotřeba (m³) | `consumption_m3` | `consumption_m3` |
| spotreba_mwh | Spotřeba (MWh) | `consumption_mwh` | `consumption_mwh` |
| koef_prepoctu | Přepočtový koeficient | `conversion_factor` | `conversion_factor` |
| spalne_teplo | Spalné teplo (MJ/m³) | `combustion_heat` | `combustion_heat` |
| mnozstvi_denni_rez_kapacity | Denní rezervovaná kapacita (m³/h) | `daily_reserved_capacity` | `daily_reserved_capacity` |
| cena_ostatni_sluzby_dodavky | Ostatní služby dodávky (Kč) | `other_supply_services_price` | `other_supply_services_price` |
| jednotkova_cena_za_obchod_rez_kapacity | Jedn. cena obch. rez. kapacity (Kč/MWh) | `trade_reserved_capacity_unit_price` | `trade_reserved_capacity_unit_price` |
| cena_za_obchod_rez_kapacity | Cena za obch. rez. kapacity (Kč) | `trade_reserved_capacity_price` | `trade_reserved_capacity_price` |
| cena_sluzby_distribuce | Cena služby distribuce (Kč) | `distribution_service_price` | `distribution_service_price` |
| jednotkova_cena_za_sluzby_distr_soustavy | Jedn. cena distribuce soustavy (Kč/MWh) | `distribution_system_unit_price` | `distribution_system_unit_price` |
| jednotkova_cena_za_distribuci_rez_kapacity | Jedn. cena distribuce rez. kapacity (Kč/m³/h) | `distribution_reserved_capacity_unit_price` | `distribution_reserved_capacity_unit_price` |
| cena_distribuce_rezervovane_kapacity | Cena distribuce rez. kapacity (Kč) | `distribution_reserved_capacity_price` | `distribution_reserved_capacity_price` |
| cena_cinnost_operatora_trhu | Cena za činnost operátora trhu (Kč) | `market_operator_price` | `market_operator_price` |
| dan_zemni_plyn_celkem | Daň ze zemního plynu celkem (Kč) | `natural_gas_tax_total` | `natural_gas_tax_total` |
| castka_bez_dph | Částka bez DPH (Kč) | `amount_ex_vat` | `amount_ex_vat` |
| castka_s_dph | Částka s DPH (Kč) | `amount_inc_vat` | `amount_inc_vat` |

---

## Teplo (teplo)

DB tabulka: `heat_details`

| České jméno | Popis | Python pole (HeatData) | DB sloupec |
|---|---|---|---|
| spotreba_gj | Spotřeba GJ (legacy) | `consumption_gj` | `consumption_gj` |
| spotreba_tepla | Spotřeba tepla (GJ) | `heat_consumption` | `heat_consumption` |
| spotreba_ohrev_tv | Spotřeba ohřev teplé vody (GJ) | `hot_water_heating` | `hot_water_heating` |
| studena_voda | Studená voda (m³) | `cold_water` | `cold_water` |
| celkova_spotreba_tepla | Celková spotřeba tepla (GJ) | `total_heat_consumption` | `total_heat_consumption` |
| rez_kapacita | Rezervovaná kapacita (kW) | `reserved_capacity` | `reserved_capacity` |
| doplnovaci_voda | Doplňovací voda (m³) | `supplementary_water` | `supplementary_water` |
| staly_mesicni_plat | Stálý měsíční plat (Kč) | `fixed_monthly_fee` | `fixed_monthly_fee` |
| variabilni_slozka | Variabilní složka (Kč) | `variable_charge` | `variable_charge` |
| castka_bez_dph | Částka bez DPH (Kč) | `amount_ex_vat` | `amount_ex_vat` |
| castka_s_dph | Částka s DPH (Kč) | `amount_inc_vat` | `amount_inc_vat` |

---

## Voda (voda)

DB tabulka: `water_details`

| České jméno | Popis | Python pole (WaterData) | DB sloupec |
|---|---|---|---|
| spotreba | Spotřeba (m³) | `consumption_m3` | `consumption_m3` |
| vodne | Vodné (Kč) | `water_rate` | `water_rate` |
| stocne | Stočné (Kč) | `sewage_rate` | `sewage_rate` |
| srazkove_vody | Srážkové vody (Kč) | `precipitation_water` | `precipitation_water` |
| odpadni_vody | Odpadní vody (Kč) | `wastewater_charge` | `wastewater_charge` |
| castka_bez_dph | Částka bez DPH (Kč) | `amount_ex_vat` | `amount_ex_vat` |
| castka_s_dph | Částka s DPH (Kč) | `amount_inc_vat` | `amount_inc_vat` |

---

## Poznámky k ground truth (CSV)

Sloupce ground truth CSV (DS1/DS2) a jejich mapování na DB:

| GT sloupec CSV | DB tabulka | DB sloupec |
|---|---|---|
| `period_from` | `invoices` | `period_from` |
| `period_to` | `invoices` | `period_to` |
| `consumption_low_tariff` | `electricity_nn_details` | `consumption_low_tariff` |
| `consumption_high_tariff` | `electricity_nn_details` | `consumption_high_tariff` |
| `total_consumption` | `electricity_nn_details` | `total_consumption` |
| `amount_ex_vat` | `invoices` | `total_amount_ex_vat` |
| `amount_inc_vat` | `invoices` | `total_amount_inc_vat` |
| `invoice_number` | `invoices` | `invoice_number` |
| `commodity` | `invoices` | `commodity` |
| `supplier` | `invoices` | `supplier_cin` (přes jméno dodavatele) |
| `due_date` | `invoices` | `due_date` |
| `issue_date` | `invoices` | `issue_date` |
| `tax_point_date` | `invoices` | `tax_point_date` |
| `supplier_tax_id` | `invoices` | `supplier_cin` |
| `customer_tax_id` | `invoices` | `customer_cin` |
| `customer_name` | `invoices` | — (v raw_extracted_json) |
| `vat_amount` | `invoices` | `vat_amount` |
| `vat_rate` | `invoices` | `vat_rate` |
| `consumption_point_code` | `invoices` | `supply_point_code` |
| `is_correction` | `invoices` | `is_correction` |
| `is_transitional` | `invoices` | `is_transitional` |
