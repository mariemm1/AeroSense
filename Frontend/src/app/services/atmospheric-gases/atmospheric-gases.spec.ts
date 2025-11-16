import { TestBed } from '@angular/core/testing';

import { AtmosphericGasesService } from './atmospheric-gases';

describe('AtmosphericGases', () => {
  let service: AtmosphericGasesService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AtmosphericGasesService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
