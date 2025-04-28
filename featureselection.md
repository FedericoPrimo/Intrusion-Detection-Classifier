| Feature                 | MI Score   | Commento                                                                                                      |
| ----------------------- | ---------- | ------------------------------------------------------------------------------------------------------------- |
| `Name`                | 1.097      | ⚠️ Va esclusa — identificativo univoco,**leakage totale**                                            |
| `Billing Amount`      | 1.097      | 🔥 Molto informativa! → può riflettere**intensità o durata**                                         |
| `Test Results`        | 1.097      | ✅ Altamente correlata all’esito → tiene!                                                                   |
| `Doctor`,`Hospital` | ~1.08-1.09 | ⚠️ Attenzione: potrebbero essere**proxy dell’outcome**(es. certi dottori usano sempre certe terapie) |
| `Room Number`         | 0.45       | 🟡 Curiosa — forse associata a tipologie di ricovero                                                         |
| `Age`                 | 0.07       | ✅ Tiene! Poco ma clinicamente rilevante                                                                      |
| `Length of Stay`      | 0.03       | ✅ Debole ma utile                                                                                            |
| `Insurance Provider`  | 0.007      | ❌ Probabilmente trascurabile                                                                                 |
| `Medical Condition`   | 0.005      | ❌ Così com'è → probabilmente troppo dispersa (vedi sotto)                                                 |
| `Blood Type`          | 0.004      | ❌ Trascurabile                                                                                               |
| `Admission Type`      | 0.004      | ❌ Ma potenzialmente utile se combinata ad altre                                                              |
| `Gender`              | 0.003      | ❌ Forse irrilevante da sola, ma combinata con età?                                                          |
| `Medication`          | 0.001      | ❌ Ma… dipende da com’è rappresentata! Vedi sotto.                                                         |
