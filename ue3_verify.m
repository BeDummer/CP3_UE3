% CG Methode zur Loesung einer Laplacegleichung der Form Ax = b

% definiere Grid Gr��e

N=8;

% Aufrufen der laplace.m Routine
% Matrix A wird erhalten

A = laplace(N);

% initialisiere Bestimmungsvektor b aus random Vektor v in cg.c
b = [10.30 19.80 10.50 11.50 8.10 25.50 7.40 23.60 4.10 20.50 18.60 17.10 24.20 25.10 22.70 7.00 12.40 19.40 8.40 24.80 2.70 23.20 23.10 14.10 11.80 9.00 4.60 9.90 5.10 15.90 20.10 15.40 10.20 5.00 1.30 18.30 4.90 8.80 16.30 9.00 3.70 9.30 0.50 2.30 8.80 23.30 9.40 21.20 17.10 17.80 20.50 19.80 15.50 18.00 8.40 1.70 1.40 13.00 11.60 6.50 3.30 6.10 22.00 13.50];                                                                                                                                                
b = b';        

% initialisiere Loesungsvektor x

x0 = zeros(N^2,1);

% Residuum r0 definieren
% Toleranz tol fuer Quadrat der Residuumsnorm
% Norm des initialisierten Residuums r0initnorm

r0 = b;
tol = 10^(-6);
r0initnorm = norm(r0);

if r0'*r0 < tol         % falls schon gew�nschte tol erreicht, dann:
    fprintf('ich habe fertig! x = x0\n')    % Ausgabetext
    return;
end


% Initialisierung p0, Iterationsschritt k und Iterationsmaximum kmax
% Zaehlung der Iterationsschritte in kcount
% Werte xk f�r rel. Fehlers speichern in xkcount pro Iteration
% relerr=xk-x0 in relerrcount; Norm dessen in relerrcountnorm
% Werte des Residuums (Norm) speichern in r0normcount pro Iteration
% Speicherplatz somit vorab belegt

p0 = r0;
k = 0;
kmax = 10000;



while k < kmax
    
    kcount(1,k+1)=k;                    % k-ten Iterationsschritt zaehlen
    
    s = A * p0;                         % vorgegebene Iteration implementiert
    alpha = (p0'*r0)/(p0'*s);
    xkcount(:,k+1) = x0;
    x0 = x0 + alpha * p0;
    r0alt = r0;
    r0 = r0 - alpha * s;
 
    
    if r0'*r0 < tol                             % falls schon gew�nschte tol erreicht, dann:
     
        fprintf('ich habe fertig! x = x0\n')    % Ausgabetext
        x0
        break
    end
    
    beta = (r0' * r0)/(r0alt' * r0alt); % restliche Iteration falls tol noch nicht erreicht
    p0 = r0 + beta * p0;
    
    
    
    k = k+1;
    
end


fprintf('ich habe fertig! x = x0\n')    % Ausgabetext
x0

